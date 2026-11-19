"""Session loop: planner -> checks -> trials -> evaluator -> keep rule -> log -> commit. LearnSession: the learnings
run (--mode learn), one hypothesis per batch, R starts per sensor_layout, verdicts, learnings, no-gain stop."""

from __future__ import annotations

import json
import math
import platform
import secrets
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

from heat_inverse import config

from . import evaluator, hypothesis, layout, learnings, log, noise, state, trial
from . import problem as P

EVALUATOR_FILES = ("src/autoresearch/problem.py", "src/autoresearch/evaluator.py", "src/heat_inverse")
SCREEN = "x0-x1=0; x1-z0=0; z0-z1=0"  # design C3 and Y1: equal side counts


def run_id(prefix: str) -> str:
    """<prefix>_<local time stamp>_<4 hex digits>."""
    return f"{prefix}_{datetime.now():%Y%m%dT%H%M%S}_{secrets.token_hex(2)}"


def git(root, *args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=check)
    return r.stdout.strip()


def is_git_repo(root) -> bool:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--git-dir"], capture_output=True).returncode == 0


def has_commits(root) -> bool:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"],
                          capture_output=True).returncode == 0


def evaluator_hash(code_root) -> str:
    """Last commit that changed the evaluator files; 'none' outside a git repository or before its first commit."""
    if not (is_git_repo(code_root) and has_commits(code_root)):
        return "none"
    return git(code_root, "log", "-1", "--format=%H", "--", *EVALUATOR_FILES) or "none"


def src_dirty(code_root) -> bool:
    """True if src/ has uncommitted changes, or if the code is not in a git repository."""
    return not is_git_repo(code_root) or bool(git(code_root, "status", "--porcelain", "--", "src"))


def commit(root, paths, message):
    git(root, "add", *[str(p) for p in paths])
    if subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet"]).returncode != 0:
        git(root, "commit", "-q", "-m", message)


def no_gain(counter: int, best_before, best_after, eps_eff: float) -> int:
    """No-gain counter after one batch (design Z1): 0 if the best score fell by more than ε_eff, else counter + 1;
    unchanged while there is no earlier valid score."""
    if best_before is None:
        return counter
    if best_after is not None and best_before - best_after > eps_eff:
        return 0
    return counter + 1


class Session:
    def __init__(self, *, session_id, mode, n, b, m, maxiter, kill_time, budget_s, mirror_reduce, planner,
                 root=config.PROJECT_ROOT, out_root=config.OUTPUTS, commit_mode="batch", resume=False,
                 allow_dirty=False, banks=None, bank_file=P.BANK_FILE, snapshot_iter=None, max_empty_calls=3,
                 max_batches=None, settings=None, seeds=None):
        self.id, self.mode, self.n, self.b, self.m = session_id, mode, n, b, m
        self.maxiter, self.kill_time, self.budget_s = maxiter, kill_time, budget_s
        self.mirror_reduce, self.planner = mirror_reduce, planner
        self.seeds = None if seeds is None else tuple(int(s) for s in seeds)  # None: v1 start rule, one start
        self.root, self.commit_mode, self.resume = Path(root), commit_mode, resume
        self.allow_dirty, self.banks, self.bank_file = allow_dirty, banks, Path(bank_file)
        self.snapshot_iter, self.max_empty_calls, self.max_batches = snapshot_iter, max_empty_calls, max_batches
        self.log_dir = self.root / "sessions" / session_id
        self.out_dir = Path(out_root) / session_id
        self.log = log.Log(self.log_dir)
        self.settings = dict(settings or {})

    # ------------------------------------------------------------------ setup

    def _prepare(self):
        if jax.devices()[0].platform != "gpu":
            raise RuntimeError("JAX does not run on the GPU")
        code_root = config.PROJECT_ROOT
        dirty = src_dirty(code_root)
        if dirty and not self.allow_dirty:
            raise RuntimeError("src/ has uncommitted changes, or the code is not in a git repository")
        if self.log_dir.exists() and not self.resume:
            raise FileExistsError(f"{self.log_dir} exists; a session id is used once")
        if self.commit_mode != "none" and not is_git_repo(self.root):
            raise RuntimeError(f"{self.root} is not a git repository")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "calls").mkdir(parents=True, exist_ok=True)
        self.B_train, self.B_val = self.banks if self.banks is not None else P.load_banks(self.bank_file)
        self.ev_hash = evaluator_hash(code_root)
        self.total = len(layout.enumerate_layouts(self.n, self.mirror_reduce))
        in_git = is_git_repo(code_root) and has_commits(code_root)
        self.meta = dict(
            self.settings, session_id=self.id, mode=self.mode, N=self.n, B=self.b, M=self.m,
            maxiter=self.maxiter, kill_time_s=self.kill_time, budget_s=self.budget_s,
            mirror_reduce=self.mirror_reduce, n_sensor_layouts=self.total, ftol=trial.FTOL, gtol=trial.GTOL,
            seeds=list(self.seeds) if self.seeds is not None else "v1 start rule",
            d_min_m=P.D_MIN, tol_spacing_m=evaluator.TOL_SPACING, margins_lo_m=P.M_LO.tolist(),
            margins_hi_m=P.M_HI.tolist(), mu=P.MU, faces=P.FACES, bank_file=str(self.bank_file),
            bank_sha256=P.file_sha256(self.bank_file) if self.bank_file.exists() else "none",
            bank_key=P.BANK_KEY, evaluator_hash=self.ev_hash,
            git_head=git(code_root, "rev-parse", "HEAD", check=False) if in_git else "none",
            git_branch=git(code_root, "rev-parse", "--abbrev-ref", "HEAD", check=False) if in_git else "none",
            src_dirty=dirty, jax=jax.__version__, devices=[str(d) for d in jax.devices()],
            python=platform.python_version(), started=datetime.now(timezone.utc).isoformat(),
        )
        self._write_meta()
        self.rows = self.log.results()
        self.cache = log.build_cache(self.rows, self.mirror_reduce)
        notes_file = self.out_dir / "notes.json"
        self.notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
        keeps = [r["score"] for r in self.rows if r["status"] == "keep"]
        self.best = min(keeps) if keeps else None
        done = [int(r["id"].split("-")[0][1:]) for r in self.rows]
        self.batch = max(done) + 1 if done else 1

    def _write_meta(self):
        (self.log_dir / "session.json").write_text(json.dumps(self.meta, indent=1, default=str))

    # ------------------------------------------------------------------ one trial

    def run_starts(self, counts, seeds, out_dir: Path):
        """The starts of one sensor_layout, one after another (design J9 a), each with its own kill time; the hard
        spacing check on each start before the selection of the valid start with the lowest Φ_train (J3, J4).
        -> (summary of the selected start, starts.csv rows without id). With no valid start the selection falls back
        to the scored starts, so that the log row still shows a value; it is then marked infeasible."""
        t0 = time.perf_counter()
        recs = []
        for seed in seeds:
            ts = time.perf_counter()
            X0, faces = layout.start_design(counts, seed=seed)
            res = ev = pt = err = None
            try:
                res = trial.optimise(X0, faces, self.B_train, self.maxiter, kill_time=self.kill_time)
                if not res.killed:
                    ev = evaluator.evaluate(res.X, faces, self.B_val)
                    pt = trial.phi_train(res.X, self.B_train)
            except Exception:  # a failed start counts as a crashed start; the other starts go on
                err = traceback.format_exc()
            recs.append(dict(seed=seed, X0=X0, faces=faces, res=res, ev=ev, pt=pt, err=err,
                             wall=time.perf_counter() - ts))
        scored_ = [i for i, r in enumerate(recs) if r["err"] is None and r["ev"] is not None
                   and np.isfinite(r["pt"]) and np.isfinite(r["ev"]["score"])]
        valid = [i for i in scored_ if recs[i]["ev"]["feasible"]]
        pool = valid or scored_
        sel = min(pool, key=lambda i: recs[i]["pt"]) if pool else None
        wall = time.perf_counter() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, r in enumerate(recs):
            if r["err"] is not None:
                (out_dir / f"traceback_{k}.txt").write_text(r["err"])
        N = len(recs[0]["faces"])
        nan = np.full((N, 3), np.nan)
        L = max([len(r["res"].loss) for r in recs if r["res"] is not None], default=0)
        pad = lambda a: np.pad(np.asarray(a, float), (0, L - len(a)), constant_values=np.nan)  # noqa: E731
        res_or = lambda r, f, d: getattr(r["res"], f) if r["res"] is not None else d  # noqa: E731
        X_start_all = np.stack([res_or(r, "X_start", r["X0"]) for r in recs])
        X_all = np.stack([res_or(r, "X", nan) for r in recs])
        np.savez(out_dir / "trial.npz", X_start=X_start_all[sel if sel is not None else 0],
                 X=X_all[sel] if sel is not None else nan, faces=recs[0]["faces"], seeds=np.array(seeds),
                 selected=-1 if sel is None else sel, X_start_all=X_start_all, X_all=X_all,
                 loss=np.stack([pad(res_or(r, "loss", [])) for r in recs]),
                 t_iter=np.stack([pad(res_or(r, "t_iter", [])) for r in recs]),
                 nit=np.array([res_or(r, "nit", -1) for r in recs]),
                 converged=np.array([bool(res_or(r, "converged", False)) for r in recs]),
                 killed=np.array([bool(res_or(r, "killed", False)) for r in recs]),
                 crashed=np.array([i not in scored_ for i in range(len(recs))]),
                 phi_train=np.array([r["pt"] if r["pt"] is not None else np.nan for r in recs]),
                 phi_val=np.array([r["ev"]["score"] if r["ev"] is not None else np.nan for r in recs]),
                 valid=np.array([i in valid for i in range(len(recs))]),
                 wall_start=np.array([r["wall"] for r in recs]), wall_total=wall)
        rows = []
        for k, r in enumerate(recs):
            ev, res = r["ev"], r["res"]
            rows.append(dict(start=k, seed=r["seed"], phi_train=r["pt"] if r["pt"] is not None else "",
                             phi_val=ev["score"] if ev else "", s1=ev["s1"] if ev else "", s2=ev["s2"] if ev else "",
                             rho=ev["rho"] if ev else "", n_iter=res.nit if res is not None else "",
                             converged=bool(res.converged) if res is not None else False,
                             min_spacing_mm=ev["min_spacing_mm"] if ev else "", valid=k in valid,
                             selected=k == sel, wall_time=r["wall"]))
        return dict(rec=recs[sel] if sel is not None else None, n_valid=len(valid), wall=wall), rows

    def _trial_starts(self, tid, counts, description) -> dict:
        """One trial with R starts (design J6, J7): keep rule on the selected start; crash if no start has a score,
        discard (infeasible) if no start is valid."""
        summ, starts = self.run_starts(counts, self.seeds, self.out_dir / tid)
        rec, note = summ["rec"], None
        if rec is None:
            status, note = "crash", "all starts crashed"
        elif summ["n_valid"] == 0:
            status, note = "discard", "infeasible"
        elif self.best is None or rec["ev"]["score"] < self.best:
            status = "keep"
            self.best = rec["ev"]["score"]
        else:
            status = "discard"
        ev, res = (rec["ev"], rec["res"]) if rec is not None else ({}, None)
        row = dict(id=tid, config=layout.fmt(counts), score=ev.get("score", ""), s1=ev.get("s1", ""),
                   s2=ev.get("s2", ""), rho=ev.get("rho", ""), n_iter=res.nit if res is not None else "",
                   converged=bool(res.converged) if res is not None else False, wall_time=summ["wall"],
                   evaluator_hash=self.ev_hash, status=status, description=description)
        self.log.append_starts([dict(s, id=tid) for s in starts])
        if note is not None:
            self.notes[tid] = note
            (self.out_dir / "notes.json").write_text(json.dumps(self.notes, indent=1))
        return row

    def run_trial(self, tid, counts, description) -> dict:
        if self.seeds is not None:
            return self._trial_starts(tid, counts, description)
        t0 = time.perf_counter()
        X0, faces = layout.start_design(counts)
        res = ev = err = None
        try:
            res = trial.optimise(X0, faces, self.B_train, self.maxiter, kill_time=self.kill_time,
                                 snapshot_iter=self.snapshot_iter)
            ev = evaluator.evaluate(res.X, faces, self.B_val)
        except Exception:  # a failed trial is logged as crash, the session goes on
            err = traceback.format_exc()
        wall = time.perf_counter() - t0
        note = None
        if err is not None or res.killed:
            status = "crash"
            note = "killed" if err is None else "exception"
        elif not ev["feasible"]:
            status, note = "discard", "infeasible"
        elif self.best is None or ev["score"] < self.best:
            status = "keep"
            self.best = ev["score"]
        else:
            status = "discard"
        scored = status != "crash"
        row = dict(id=tid, config=layout.fmt(counts), score=ev["score"] if scored else "",
                   s1=ev["s1"] if scored else "", s2=ev["s2"] if scored else "",
                   rho=ev["rho"] if scored else "", n_iter=res.nit if res is not None else "",
                   converged=res.converged if res is not None else False, wall_time=wall,
                   evaluator_hash=self.ev_hash, status=status, description=description)
        d = self.out_dir / tid
        d.mkdir(parents=True, exist_ok=True)
        if err is not None:
            (d / "traceback.txt").write_text(err)
        if res is not None:
            snap = res.X_snapshot
            np.savez(d / "trial.npz", X_start=res.X_start, X=res.X, faces=res.faces, loss=res.loss,
                     t_iter=res.t_iter, nit=res.nit, converged=res.converged, killed=res.killed,
                     message=res.message, wall_optimise=res.wall_time, wall_total=wall,
                     X_snapshot=snap if snap is not None else np.zeros((0, 3)),
                     score_snapshot=evaluator.evaluate(snap, faces, self.B_val)["score"] if snap is not None
                     else np.nan,
                     **({k: v for k, v in ev.items()} if ev is not None else {}))
        if note is not None:
            self.notes[tid] = note
            (self.out_dir / "notes.json").write_text(json.dumps(self.notes, indent=1))
        return row

    # ------------------------------------------------------------------ loop

    def _check(self, proposals):
        """Split proposals into runnable (counts, description) and marks for the next state."""
        runnable, marks, seen = [], [], set()
        for i, (config_str, desc) in enumerate(proposals):
            if i >= self.b:
                marks.append(f"ignored {config_str}: more than B = {self.b} proposals")
                continue
            try:
                counts = layout.parse(config_str, self.n)
            except ValueError as e:
                marks.append(f"rejected {config_str!r}: {e}")
                continue
            key = layout.canonical(counts) if self.mirror_reduce else counts
            if key in self.cache:
                same = "" if layout.fmt(key) == layout.fmt(counts) else " (mirror image)"
                marks.append(f"repeat {config_str} = {self.cache[key]['id']}{same}: not run")
                continue
            if key in seen:
                marks.append(f"repeat {config_str} within the same reply: not run")
                continue
            seen.add(key)
            desc = " ".join(str(desc).split())
            if len(desc) > 100:
                marks.append(f"description of {config_str} cut to 100 characters")
                desc = desc[:100]
            runnable.append((counts, desc))
        return runnable, marks

    def run(self) -> dict:
        self._prepare()
        t_start = time.perf_counter()
        marks, empty, stop, n_batches = [], 0, None, 0
        try:
            while stop is None:
                elapsed = time.perf_counter() - t_start
                untried = self.total - len(self.cache)
                if elapsed >= self.budget_s:
                    stop = "budget"
                    break
                if self.max_batches is not None and n_batches >= self.max_batches:
                    stop = "max batches"
                    break
                n_batches += 1
                if untried == 0:
                    stop = "all sensor_layouts tried"
                    break
                call_id = f"c{self.batch:03d}"
                st = state.build(self.rows, self.notes, n=self.n, b=self.b, m=self.m,
                                 remaining_s=self.budget_s - elapsed if math.isfinite(self.budget_s) else 0.0,
                                 untried=untried, marks=marks)
                reply = self.planner.propose(st, call_id)
                if reply.call is not None:
                    self.log.append_call(reply.call)
                if reply.raw is not None:
                    reply.raw["state"] = st
                    (self.out_dir / "calls" / f"{call_id}.json").write_text(
                        json.dumps(reply.raw, indent=1, default=str))
                runnable, marks = self._check(reply.proposals)
                if not runnable:
                    empty += 1
                    if empty >= self.max_empty_calls:
                        stop = f"no runnable proposal in {empty} calls"
                    self.batch += 1
                    continue
                empty = 0
                for k, (counts, desc) in enumerate(runnable, 1):
                    if time.perf_counter() - t_start >= self.budget_s:
                        stop = "budget"
                        break
                    row = self.run_trial(f"b{self.batch:03d}-{k:02d}", counts, desc)
                    self.log.append_result(row)
                    parsed = log.parse_result({k2: str(v) for k2, v in row.items()})
                    self.rows.append(parsed)
                    self.cache[layout.canonical(counts) if self.mirror_reduce else counts] = parsed
                if self.commit_mode == "batch":
                    commit(self.root, [self.log_dir], f"{self.id}: batch {self.batch:03d}")
                self.batch += 1
        finally:
            calls = self.log.calls()
            keeps = [r for r in self.rows if r["status"] == "keep"]
            best = min(keeps, key=lambda r: r["score"]) if keeps else None
            self.meta.update(stop=stop or "interrupted", elapsed_s=time.perf_counter() - t_start,
                             n_trials=len(self.rows), n_calls=len(calls),
                             cost_usd=sum(float(c["cost"]) for c in calls),
                             best_id=best["id"] if best else None, best_config=best["config"] if best else None,
                             best_score=best["score"] if best else None,
                             finished=datetime.now(timezone.utc).isoformat())
            self._write_meta()
            if self.commit_mode in ("batch", "end"):
                commit(self.root, [self.log_dir], f"{self.id}: end ({self.meta['stop']})")
        return self.meta


class LearnSession(Session):
    """--mode learn (project_docs/case2_dopt_autoresearch_learnings_design.md): batch 1 runs the screening
    constraint; each later batch runs the untried members of S(h) of one checked hypothesis; the harness computes the
    verdict, the group counters, promotions and flags; stop on K batches with no gain > ε_eff, time, nothing runnable
    or all sensor_layouts tried; then the top sensor_layouts run again with R new seeds."""

    def __init__(self, *, noise_cal: dict, b_max: int, k_stop: int, n_promote: int, screen: str = SCREEN,
                 n_confirm: int = 5, max_retries: int = 2, **kw):
        super().__init__(mode="learn", b=b_max, mirror_reduce=True, seeds=tuple(range(noise_cal["R"])), **kw)
        self.cal, self.k_stop, self.n_promote = noise_cal, k_stop, n_promote
        self.screen, self.n_confirm, self.max_retries = screen, n_confirm, max_retries

    def _prepare(self):
        super()._prepare()
        if self.rows:
            raise RuntimeError("--mode learn does not resume a session")
        c = self.cal
        self.meta.update(R=c["R"], sigma_R=c["sigma_R"], tau=c["tau"], eps=c["eps"], eps_eff=c["eps_eff"],
                         K=self.k_stop, B_max=self.b, n_promote=self.n_promote, calibration_id=c["id"],
                         calibration_file=c["file"], screen=hypothesis.fmt(hypothesis.parse(self.screen)),
                         n_confirm=self.n_confirm, confirm_seeds=list(range(c["R"], 2 * c["R"])),
                         max_retries=self.max_retries, max_short=learnings.MAX_SHORT,
                         max_long=learnings.MAX_LONG)
        self._write_meta()
        self.hyps, self.L, self.report, self.n_empty = {}, learnings.Learnings(self.n), [], 0

    # ------------------------------------------------------------------ helpers

    def _valid(self) -> list[dict]:
        return state.scored(self.rows, self.notes)

    def _best(self) -> dict | None:
        v = self._valid()
        return min(v, key=lambda r: r["score"]) if v else None

    def _default_ref(self, step: str, parent: str) -> str:
        """Best trial of the parent; for a root, or a parent with no valid member, the best trial so far (Y11)."""
        if step != "root" and parent in self.hyps and self.hyps[parent]["best_id"]:
            return self.hyps[parent]["best_id"]
        best = self._best()
        return best["id"] if best else ""

    def _state(self, remaining_s: float, marks) -> str:
        return state.build_learn(
            self.rows, self.notes, list(self.hyps.values()), self.L.level("long"), self.L.recent_short(),
            len(self.L.level("short")), self.report, n=self.n, r=self.cal["R"], sigma=self.cal["sigma_R"],
            tau=self.cal["tau"], b_max=self.b, n_promote=self.n_promote, m=self.m,
            remaining_s=remaining_s, untried=self.total - len(self.cache), total=self.total, marks=marks)

    # ------------------------------------------------------------------ reply checks (design X6–X7, Y3–Y6, M9–M10)

    def _check_hypothesis(self, h: dict, decisions):
        errs = []
        step, pred, parent = h["step"].strip(), h["prediction"].strip(), h["parent_id"].strip()
        if step not in hypothesis.STEPS:
            errs.append(f"step {step!r}: use one of {', '.join(hypothesis.STEPS)}")
        if pred not in hypothesis.PREDICTIONS:
            errs.append(f"prediction {pred!r}: use one of {', '.join(hypothesis.PREDICTIONS)}")
        if step == "root":
            parent = ""
        elif parent not in self.hyps:
            errs.append(f"parent_id {parent!r}: no hypothesis has this id (step {step} needs a parent)")
        try:
            region = hypothesis.parse(h["region"])
        except ValueError as e:
            return errs + [f"region {h['region']!r}: {e}"], None
        try:
            context = hypothesis.parse(h["context"]) if h["context"].strip() else ()
        except ValueError as e:
            return errs + [f"context {h['context']!r}: {e}"], None
        rtext, ctext = hypothesis.fmt(region), hypothesis.fmt(context)
        where = f"region {rtext}; context {ctext or 'none'}"
        S = hypothesis.members(region + context, self.n)
        new = [u for u in S if u not in self.cache]
        if not S:
            errs.append(f"{where}: S(h) is empty at N = {self.n}")
        elif not new:
            errs.append(f"{where}: all {len(S)} members of S(h) are tried; S(h) needs at least one untried "
                        "sensor_layout")
        elif len(new) > self.b:
            errs.append(f"{where}: S(h) has {len(new)} untried members ({len(S)} in total), more than "
                        f"B_max = {self.b}; add a condition to the context")
        ref = h["ref_id"].strip() or self._default_ref(step, parent)
        if ref not in {r["id"] for r in self._valid()}:
            errs.append(f"ref_id {ref!r}: not a trial with a valid score")
        demoted = {d["rule_id"].strip() for d in decisions if d["decision"] == "demote"}
        rules = [g["group_id"] for g in self.L.level("long") if g["group_id"] not in demoted]
        own = self.L.find(rtext, pred) if S else None
        named = {}
        for e in h["exceptions"]:
            rid = e["rule_id"].strip()
            if rid not in rules:
                errs.append(f"exception {rid!r}: not a rule of the long-term list")
            elif not e["reason"].strip():
                errs.append(f"exception {rid}: give a reason")
            else:
                named[rid] = " ".join(e["reason"].split())
        broken = []
        for rid in rules:
            if rid == own:
                continue
            bad = [u for u in new if not self.L.obeys(rid, u)[0]]
            if bad:
                broken.append(rid)
                if rid not in named:
                    g = self.L.groups[rid]
                    errs.append(f"rule {rid} (region {g['region']}, predicted {g['prediction']}): {len(bad)} untried "
                                f"members of S(h) break it, for example {layout.fmt(bad[0])}; name the rule in "
                                "exceptions with a reason, or change the region or the context")
        plan = dict(step=step, prediction=pred, parent=parent, region=rtext, context=ctext, members=S, new=new,
                    ref=ref, exceptions={rid: named[rid] for rid in broken if rid in named},
                    rationale=" ".join(h["rationale"].split()))
        return errs, plan

    def _check_reply(self, data):
        """-> (errors, plan); the plan holds the normalised reply and S(h)."""
        try:
            decisions = [dict(rule_id=str(d["rule_id"]), decision=str(d["decision"]), reason=str(d["reason"]))
                         for d in data["flag_decisions"]]
            hr = data["hypothesis"]
            h = {k: str(hr[k]) for k in ("parent_id", "step", "region", "context", "prediction", "ref_id",
                                         "rationale")}
            h["exceptions"] = [dict(rule_id=str(e["rule_id"]), reason=str(e["reason"])) for e in hr["exceptions"]]
        except (KeyError, TypeError) as e:
            return [f"the reply does not follow the reply format: missing or wrong field {e}"], None
        errs = self.L.check_decisions(decisions)
        herrs, plan = self._check_hypothesis(h, decisions)
        if plan is not None:
            plan["decisions"] = decisions
        return errs + herrs, plan

    def _ask(self, call_id: str, remaining_s: float, marks):
        """One planner call with at most max_retries corrected replies (X6) -> (plan or None, marks for the next
        state)."""
        st = self._state(remaining_s, marks)
        messages, errs = [{"role": "user", "content": st}], []
        for retry in range(self.max_retries + 1):
            reply = self.planner.respond(messages, call_id, retry)
            self.log.append_call(reply.call)
            plan = None
            if reply.data is None:
                errs = [reply.error]
            else:
                errs, plan = self._check_reply(reply.data)
            audit = dict(reply.raw or {}, errors=errs, messages=messages)
            (self.out_dir / "calls" / f"{call_id}_r{retry}.json").write_text(json.dumps(audit, indent=1, default=str))
            if not errs:
                return plan, []
            if reply.content is None:
                break
            messages = messages + [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": "The harness rejected your reply. Errors:\n"
                                            + "\n".join(f"- {e}" for e in errs)
                                            + "\nSend a full corrected reply (flag_decisions, hypothesis)."}]
        return None, [f"Call {call_id}: nothing runnable after {retry + 1} replies. Errors of the last reply:",
                      *(f"- {e}" for e in errs)]

    # ------------------------------------------------------------------ one batch

    def _screen_plan(self) -> dict:
        con = hypothesis.parse(self.screen)
        S = hypothesis.members(con, self.n)
        return dict(step="root", prediction="", parent="", region=hypothesis.fmt(con), context="", members=S,
                    new=[u for u in S if u not in self.cache], ref="", exceptions={},
                    rationale="screening batch (design Y1)", decisions=[])

    def _accept(self, plan: dict, hid: str) -> dict:
        """Apply an accepted reply: flag decisions, exceptions, closed branches, group of the hypothesis; -> hypothesis
        record."""
        self.L.apply_decisions(plan["decisions"], hid)
        lines = [f"Flag decision for {d['rule_id']}: {d['decision']}." for d in plan["decisions"]]
        for rid in plan["exceptions"]:
            self.L.record_exception(rid, hid)
        if plan["step"] == "back" and self.hyps:
            closed = hypothesis.path_to(self.hyps, list(self.hyps)[-1], plan["parent"])
            for k in closed:
                self.hyps[k]["branch"] = "closed"
            if closed:
                lines.append(f"Branch closed by this back step: {' '.join(closed)}.")
        S = plan["members"]
        hyp = dict(hyp_id=hid, parent_id=plan["parent"], step=plan["step"], region=plan["region"],
                   context=plan["context"], prediction=plan["prediction"], ref_id=plan["ref"], n_set=len(S),
                   B_h=len(plan["new"]),
                   reused_ids=" ".join(self.cache[u]["id"] for u in S if u in self.cache), best_id="", n_cmp=0,
                   n_ref=hypothesis.selection_size(self.hyps, plan["ref"]) if plan["ref"] else 0, delta=None,
                   tau_h=None, verdict="", branch="open",
                   exceptions="; ".join(f"{k}: {v}" for k, v in plan["exceptions"].items()),
                   rationale=plan["rationale"])
        self.hyps[hid] = hyp
        self._group = self.L.assign(plan["region"], plan["prediction"], hid) if plan["ref"] else None
        self._accept_lines = lines
        return hyp

    def _run_batch(self, hyp: dict, new, t_start: float):
        """Trials of the untried members of S(h) -> (rows, True if the time budget ran out during the batch)."""
        rows = []
        desc = f"{hyp['hyp_id']}: {hyp['region']}" + (f"; {hyp['context']}" if hyp["context"] else "")
        desc = desc[:100]
        for k, counts in enumerate(new, 1):
            if time.perf_counter() - t_start >= self.budget_s:
                return rows, True
            row = self.run_trial(f"{hyp['hyp_id']}-{k:02d}", counts, desc)
            self.log.append_result(row)
            parsed = log.parse_result({k2: str(v) for k2, v in row.items()})
            self.rows.append(parsed)
            self.cache[layout.canonical(counts)] = parsed
            rows.append(parsed)
        return rows, False

    def _finish(self, hyp: dict, members, batch_rows):
        """Δ, τ_h and verdict (Y11–Y14), group counters, promotion and flags (M6–M8); the report of the batch."""
        c = self.cal
        valid_ids = {r["id"] for r in self._valid()}
        by_id = {r["id"]: r for r in self.rows}
        cmp = [self.cache[u] for u in members if u in self.cache and self.cache[u]["id"] in valid_ids
               and self.cache[u]["id"] != hyp["ref_id"]]
        best = min(cmp, key=lambda r: r["score"]) if cmp else None
        hyp.update(n_cmp=len(cmp), best_id=best["id"] if best else "")
        lines = [f"Hypothesis {hyp['hyp_id']}: {len(batch_rows)} sensor_layouts run, {hyp['n_set'] - hyp['B_h']} "
                 "reused from the log.", *self._accept_lines]
        if hyp["ref_id"]:
            ref = by_id[hyp["ref_id"]]
            th = hypothesis.tau_h(c["tau"], c["sigma_R"], len(cmp), hyp["n_ref"])
            delta = best["score"] - ref["score"] if best else None
            hyp.update(delta=delta, tau_h=th, verdict=hypothesis.verdict(delta, hyp["prediction"], th))
            corr = max(0.0, noise.expected_max(len(cmp)) - noise.expected_max(hyp["n_ref"]))
            lines += [state.hypothesis_line(hyp, by_id),
                      f"Δ = {state.show('delta', delta)}; τ_h = τ + max(0, c({len(cmp)}) − c({hyp['n_ref']})) σ_R = "
                      f"{c['tau']:.4f} + {corr:.3f} × {c['sigma_R']:.4f} = {th:.4f} -> {hyp['verdict']}."]
            self.L.record_test(self._group, hyp["hyp_id"], hyp["verdict"])
            g = self.L.groups[self._group]
            lines.append(f"Group {g['group_id']} (region {g['region']}, predicted {g['prediction']}): {g['status']}; "
                         f"tests supported/refuted/inconclusive {g['n_support']}/{g['n_refute']}/"
                         f"{g['n_inconclusive']}.")
            lines += self.L.promote(self._group, self.n_promote, hyp["hyp_id"])
        else:
            lines += [state.hypothesis_line(hyp, by_id), "Screening batch: no verdict."]
        lines += self.L.update_flags(hyp, self._group, batch_rows, self._valid(), c["tau"])
        b = self._best()
        lines.append(f"Best score so far: {b['id']} {b['config']} {state.show('score', b['score'])}." if b
                     else "No valid trial so far.")
        self.report = lines
        self.log.write_hypotheses(list(self.hyps.values()))
        self.log.write_learnings(self.L.rows(), learnings.GROUP_COLUMNS)

    def confirm(self) -> list[dict]:
        """Design Z6: the n_confirm best valid sensor_layouts again with seeds R…2R−1."""
        R, out = self.cal["R"], []
        for r in sorted(self._valid(), key=lambda x: x["score"])[:self.n_confirm]:
            summ, starts = self.run_starts(layout.parse(r["config"]), tuple(range(R, 2 * R)),
                                           self.out_dir / "confirm" / r["id"])
            self.log.append_starts([dict(s, id=r["id"]) for s in starts], confirm=True)
            ok = summ["rec"] is not None and summ["n_valid"] > 0
            ev = summ["rec"]["ev"] if ok else {}
            row = dict(id=r["id"], config=r["config"], score_session=r["score"], score_confirm=ev.get("score", ""),
                       phi_train_confirm=summ["rec"]["pt"] if ok else "", s1=ev.get("s1", ""), s2=ev.get("s2", ""),
                       rho=ev.get("rho", ""), n_valid=summ["n_valid"], wall_time=summ["wall"])
            self.log.append_confirm(row)
            out.append(row)
        return out

    # ------------------------------------------------------------------ loop

    def run(self) -> dict:
        self._prepare()
        t_start = time.perf_counter()
        marks, empty, n_gain, stop, n_batches, confirmed, t_confirm = [], 0, 0, None, 0, [], 0.0
        try:
            while stop is None:
                elapsed = time.perf_counter() - t_start
                if elapsed >= self.budget_s:
                    stop = "budget"
                    break
                if self.max_batches is not None and n_batches >= self.max_batches:
                    stop = "max batches"
                    break
                if self.total == len(self.cache):
                    stop = "all sensor_layouts tried"
                    break
                n_batches += 1
                hid = f"b{self.batch:03d}"
                if not self.hyps:
                    plan = self._screen_plan()
                else:
                    remaining = self.budget_s - elapsed if math.isfinite(self.budget_s) else 0.0
                    plan, marks = self._ask(f"c{self.batch:03d}", remaining, marks)
                    if plan is None:
                        empty += 1
                        self.n_empty += 1
                        if empty >= self.max_empty_calls:
                            stop = f"nothing runnable in {empty} calls"
                        self.batch += 1
                        continue
                    empty = 0
                b0 = self._best()
                hyp = self._accept(plan, hid)
                batch_rows, over = self._run_batch(hyp, plan["new"], t_start)
                self._finish(hyp, plan["members"], batch_rows)
                b1 = self._best()
                n_gain = no_gain(n_gain, b0["score"] if b0 else None, b1["score"] if b1 else None,
                                 self.cal["eps_eff"])
                if self.commit_mode == "batch":
                    commit(self.root, [self.log_dir], f"{self.id}: batch {self.batch:03d}")
                self.batch += 1
                if over:
                    stop = "budget"
                elif n_gain >= self.k_stop:
                    stop = f"no gain > eps_eff in {n_gain} batches"
            t0 = time.perf_counter()
            confirmed = self.confirm()
            t_confirm = time.perf_counter() - t0
        finally:
            calls = self.log.calls()
            best = self._best()
            verdicts = [h["verdict"] for h in self.hyps.values() if h["verdict"]]
            self.meta.update(
                stop=stop or "interrupted", elapsed_s=time.perf_counter() - t_start, n_trials=len(self.rows),
                n_starts=len(self.rows) * len(self.seeds), n_hypotheses=len(self.hyps),
                verdicts={v: verdicts.count(v) for v in hypothesis.VERDICTS}, n_calls=len(calls),
                n_retries=sum(int(c["retry"]) > 0 for c in calls), n_nothing_runnable=self.n_empty,
                wall_llm_s=sum(float(c["wall_time"]) for c in calls),
                wall_trials_s=sum(r["wall_time"] for r in self.rows), wall_confirm_s=t_confirm,
                cost_usd=sum(float(c["cost"]) for c in calls), no_gain_batches=n_gain,
                n_rules=len(self.L.level("long")), n_groups=len(self.L.groups),
                best_id=best["id"] if best else None, best_config=best["config"] if best else None,
                best_score=best["score"] if best else None,
                confirm=[{k: r[k] for k in ("id", "config", "score_session", "score_confirm")} for r in confirmed],
                finished=datetime.now(timezone.utc).isoformat())
            self._write_meta()
            if self.commit_mode in ("batch", "end"):
                commit(self.root, [self.log_dir], f"{self.id}: end ({self.meta['stop']})")
        return self.meta
