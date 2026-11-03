"""Session loop: planner -> checks -> trials -> evaluator -> keep rule -> log -> commit."""

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

from . import evaluator, layout, log, state, trial
from . import problem as P

EVALUATOR_FILES = ("src/autoresearch/problem.py", "src/autoresearch/evaluator.py", "src/heat_inverse")


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


class Session:
    def __init__(self, *, session_id, mode, n, b, m, maxiter, kill_time, budget_s, mirror_reduce, planner,
                 root=config.PROJECT_ROOT, out_root=config.OUTPUTS, commit_mode="batch", resume=False,
                 allow_dirty=False, banks=None, bank_file=P.BANK_FILE, snapshot_iter=None, max_empty_calls=3,
                 max_batches=None, settings=None):
        self.id, self.mode, self.n, self.b, self.m = session_id, mode, n, b, m
        self.maxiter, self.kill_time, self.budget_s = maxiter, kill_time, budget_s
        self.mirror_reduce, self.planner = mirror_reduce, planner
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

    def run_trial(self, tid, counts, description) -> dict:
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
