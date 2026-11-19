"""ID case2_ar_smoke_learn: the --mode learn flow of harness.LearnSession with a fake planner (fixed replies, no API).

Context: project_docs/case2_dopt_autoresearch_learnings_design.md, §5–§8 and §11, with the claims replaced by groups
(a hypothesis has a region and a context; hypotheses with the same region set and prediction form one group). The
deterministic rules of the learnings run (region, context and S(h), reply checks, verdict, τ_h, groups, promotion,
flags, exceptions, branch closing, stop rule) are code, so they are testable without the LLM.
    A  rules, on a synthetic log at N = 25 (synthetic scores, the real check and update code of LearnSession,
       learnings.Learnings and hypothesis; no optimisation):
       S(h) of the screening constraint (7 members) and the counts 23 751 / 6825; constraint parse errors; mirror
       images in S(h); too wide a set; empty set; all members tried; context parse error; tried members reused and
       the reference excluded; Δ, τ_h and verdict; a child (same region, new context) in the same group; group key by
       set equality; promotion at n_promote = 2; broken rule with no reason, with a reason; a test of a "higher" rule
       exempt; flag by a breaking trial and by a refuted test, no flag below τ; missing flag decision; demotion; no
       promotion again without a new supported test; short-term list of the 5 latest groups; branch closing of a back
       step; c(n) against the table of design Y14; the no-gain counter (Z1); order of the state blocks (X8)
    B  end to end on the GPU, in a throwaway git repository: N = 10, R = 2 starts (seeds 0, 1), maxiter 50, B_max = 4,
       K = 2, ε = 100 (every batch has no gain): batch 1 screening (3 members); call c002: a too wide set, then a
       corrected reply (one tried member reused); call c003: a context that does not parse, then a child with the
       same region; stop after K = 2 batches; confirmation of the top 2 with seeds 2, 3. Checks the files and columns,
       the starts, the group of b002 and b003, the retries in calls.csv, the retry message, the commits, trial.npz,
       then runs plot_learn_session.py (W7–W12) and plot_session.py (W2, W5, W6) on the session.
Outputs: outputs/case2_ar_smoke_learn_<stamp>/checks.csv (case, expected, result, pass), repo/sessions/, outputs/
(plots of B in outputs/n10_learn_smoke/plots/).

Run:  scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_learn.py
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
import time

import jax
import numpy as np

from autoresearch import harness, hypothesis, layout, learnings, log, noise, planner
from autoresearch import problem as P
from heat_inverse import config

C_TABLE = {2: 0.56, 3: 0.85, 5: 1.16, 10: 1.54, 20: 1.87}  # design Y14
CAL25 = dict(id="synthetic", file="", R=8, sigma_R=0.0697, tau=float(noise.tau(0.0697)), eps=noise.EPS_GAIN,
             eps_eff=float(noise.tau(0.0697)), N=25, t_start=3.67, maxiter=5000, kill_time=60.0)
STATE_HEADERS = ("Session:", "## Long-term list", "## Tried sensor_layouts", "## Open hypotheses", "## Short-term list",
                 "## Trials", "## Last batch", "## Your previous call", "## Request")


class Checks:
    def __init__(self):
        self.rows, self.ok = [], True

    def add(self, case, expected, result, passed):
        self.ok &= bool(passed)
        self.rows.append((case, expected, result, "PASS" if passed else "FAIL"))
        print(f"{'PASS' if passed else 'FAIL'}  {case}: expected {expected}; result {result}", flush=True)

    def info(self, case, result):
        self.rows.append((case, "-", result, "INFO"))
        print(f"INFO  {case}: {result}", flush=True)


def row(tid, cfg, score):
    return dict(id=tid, config=cfg, score=score, s1=0.05, s2=0.1, rho=-0.3, n_iter=100, converged=True,
                wall_time=1.0, evaluator_hash="none", status="discard", description="synthetic")


def reply(region, context="", *, decisions=(), step="child", parent="b001", ref="", pred="lower", exceptions=()):
    return dict(flag_decisions=list(decisions),
                hypothesis=dict(parent_id=parent, step=step, region=region, context=context, prediction=pred,
                                ref_id=ref, rationale="smoke", exceptions=list(exceptions)))


def has(errs, text) -> bool:
    return any(text in e for e in errs)


def simulate(s, hid, plan, scores, default=None):
    """Accept a checked plan, log synthetic scores for its untried members, then the real after-batch update."""
    hyp = s._accept(plan, hid)
    rows = []
    for k, u in enumerate(plan["new"], 1):
        r = row(f"{hid}-{k:02d}", layout.fmt(u), scores.get(layout.fmt(u), default))
        s.rows.append(r)
        s.cache[u] = r
        rows.append(r)
    s._finish(hyp, plan["members"], rows)
    return hyp


def part_a(c, out):
    S7 = {(25 - 4 * m, m, m, m, m) for m in range(7)}
    got = set(hypothesis.members(hypothesis.parse(harness.SCREEN), 25))
    c.add("A constraint: screening set at N = 25", "7 members 25-0-0-0-0 … 1-6-6-6-6", f"{len(got)} members",
          got == S7)
    n_all, n_can = len(layout.enumerate_layouts(25, False)), len(layout.enumerate_layouts(25, True))
    c.add("A constraint: number of sensor_layouts at N = 25", "23751 / 6825", f"{n_all} / {n_can}",
          (n_all, n_can) == (23751, 6825))
    bad = ["x0+-x1=2", "y1=3", "x0<3", "x0==3", "3=3", "2x=1", "x0*2=4", ""]
    fails = []
    for t in bad:
        try:
            hypothesis.parse(t)
            fails.append(t)
        except ValueError:
            pass
    c.add("A constraint: malformed constraints raise", f"ValueError for {len(bad)}", f"accepted: {fails or 'none'}",
          not fails)
    norm = hypothesis.fmt(hypothesis.parse(" 2*y0 >= x0 + x1+z0+ z1 ;x0 = x1 "))
    c.add("A constraint: normal form", "2*y0-x0-x1-z0-z1>=0; x0-x1=0", norm, norm == "2*y0-x0-x1-z0-z1>=0; x0-x1=0")
    mir = hypothesis.members(hypothesis.parse("x1=5; x0=3"), 25)
    c.add("A constraint: mirror image in S(h)", "all members canonical with (x0, x1) = (5, 3)",
          f"{len(mir)} members", bool(mir) and all(u[1:3] == (5, 3) for u in mir))

    s = harness.LearnSession(session_id="logic", n=25, m=50, maxiter=5000, kill_time=60.0, budget_s=math.inf,
                             planner=None, noise_cal=CAL25, b_max=32, k_stop=3, n_promote=2, root=out / "logic",
                             out_root=out / "logic_out", commit_mode="none", allow_dirty=True)
    s.log_dir.mkdir(parents=True)
    scores1 = (-8.90, -9.10, -9.30, -9.40, -9.20, -9.00, -8.70)
    s.rows = [row(f"b001-{k:02d}", layout.fmt((25 - 4 * m, m, m, m, m)), sc)
              for k, (m, sc) in enumerate(zip(range(7), scores1), 1)]
    s.notes, s.cache, s.L, s.report, s.n_empty = {}, log.build_cache(s.rows, True), learnings.Learnings(25), [], 0
    s.total = n_can
    s.hyps = {"b001": dict(hyp_id="b001", parent_id="", step="root", region=hypothesis.fmt(hypothesis.parse(
        harness.SCREEN)), context="", prediction="", ref_id="", n_set=7, B_h=7, reused_ids="", best_id="b001-04",
        n_cmp=7, n_ref=0, delta=None, tau_h=None, verdict="", branch="open", exceptions="", rationale="")}

    errs, _ = s._check_reply(reply("y0>=0"))
    c.add("A too wide a set", "rejected with the size", errs[0] if errs else "accepted",
          has(errs, "more than B_max = 32"))
    errs, _ = s._check_reply(reply("y0=26"))
    c.add("A empty set", "rejected", errs[0] if errs else "accepted", has(errs, "S(h) is empty"))
    errs, _ = s._check_reply(reply(harness.SCREEN))
    c.add("A all members tried", "rejected", errs[0] if errs else "accepted", has(errs, "are tried"))
    errs, _ = s._check_reply(reply("y0=13", "x0<3"))
    c.add("A context that does not parse", "rejected", errs[0] if errs else "accepted", has(errs, "context 'x0<3'"))
    errs, plan = s._check_reply(reply("y0=13", "x0=3; x1=3"))
    c.add("A tried members: accepted, untried members only", "no error, B_h = 3 of 4",
          f"errors {errs}, B_h = {len(plan['new'])} of {len(plan['members'])}",
          not errs and (len(plan["new"]), len(plan["members"])) == (3, 4))
    h2 = simulate(s, "b002", plan, {"13-3-3-6-0": -9.70, "13-3-3-5-1": -9.45, "13-3-3-4-2": -9.42})
    c.add("A tried members: reused, reference excluded", "reused b001-04 = ref; n_h = 3; n_ref = 7",
          f"reused {h2['reused_ids']}, ref {h2['ref_id']}, n_h {h2['n_cmp']}, n_ref {h2['n_ref']}",
          (h2["reused_ids"], h2["ref_id"], h2["n_cmp"], h2["n_ref"]) == ("b001-04", "b001-04", 3, 7))
    c.add("A verdict and τ_h (c(3) < c(7): τ_h = τ)", f"Δ = -0.3000, τ_h = {CAL25['tau']:.4f}, supported",
          f"Δ = {h2['delta']:+.4f}, τ_h = {h2['tau_h']:.4f}, {h2['verdict']}",
          abs(h2["delta"] + 0.30) < 1e-12 and abs(h2["tau_h"] - CAL25["tau"]) < 1e-12 and h2["verdict"] == "supported")
    g = s.L.groups["b002"]
    c.add("A group after the first test", "b002: region y0=13, lower, supported 1/0/0, evidence b002",
          f"{g['region']}, {g['prediction']}, {g['status']} {g['n_support']}/{g['n_refute']}/{g['n_inconclusive']}, "
          f"{g['evidence']}",
          (g["region"], g["prediction"], g["status"], g["n_support"], g["evidence"])
          == ("y0=13", "lower", "supported", 1, ["b002"]))
    keys = (s.L.find("13=y0", "lower"), s.L.find("y0>=13; y0<=13", "lower"), s.L.find("y0=13", "higher"))
    c.add("A group key: same set of sensor_layouts and same prediction", "(b002, b002, None)", str(keys),
          keys == ("b002", "b002", None))

    errs, plan = s._check_reply(reply("y0=13", "x0=4; x1=2", parent="b002"))
    c.add("A child: same region, new context", "accepted", f"errors {errs}", not errs)
    h3 = simulate(s, "b003", plan, {"13-4-2-6-0": -10.00}, default=-9.50)
    th3 = hypothesis.tau_h(CAL25["tau"], CAL25["sigma_R"], 4, 3)
    c.add("A τ_h with n_h = 4 > n_ref = 3", f"{th3:.4f}, supported", f"{h3['tau_h']:.4f}, {h3['verdict']}",
          abs(h3["tau_h"] - th3) < 1e-12 and th3 > CAL25["tau"] and h3["verdict"] == "supported")
    c.add("A child in the group of its parent; promotion at n_promote = 2", "b002 evidence [b002, b003], long",
          f"{g['evidence']}, {g['level']}", g["evidence"] == ["b002", "b003"] and g["level"] == "long")

    errs, _ = s._check_reply(reply("y0=9", "x0=4; x1=4", step="root"))
    c.add("A broken rule with no reason", "rejected", errs[0] if errs else "accepted",
          has(errs, "rule b002 (region y0=13, predicted lower)"))
    exc = [dict(rule_id="b002", reason="tests y0 = 9")]
    s2 = copy.copy(s)
    s2.L = copy.deepcopy(s.L)
    s2.L.groups[s2.L.assign("y0=1", "higher", "b090")]["level"] = "long"
    errs, _ = s2._check_reply(reply("y0=1", "x0=6; x1=6", step="root", pred="higher", exceptions=exc))
    c.add("A a test of a \"higher\" rule needs no exception for it", "accepted", f"errors {errs}", not errs)
    errs, _ = s2._check_reply(reply("y0=1", "x0=6; x1=6", step="root", exceptions=exc))
    c.add("A members inside the region break a \"higher\" rule", "rejected", errs[0] if errs else "accepted",
          has(errs, "rule b090 (region y0=1, predicted higher)"))
    errs, plan = s._check_reply(reply("y0=9", "x0=4; x1=4", step="root", exceptions=exc))
    c.add("A broken rule with a reason", "accepted", f"errors {errs}", not errs)
    h4 = simulate(s, "b004", plan, {"9-4-4-8-0": -10.40}, default=-9.60)
    c.add("A exception counted", "b002 exceptions [b004]", str(g["exceptions"]), g["exceptions"] == ["b004"])
    c.add("A flag: a breaking trial beats the best obeying trial by > τ", "b002 flagged", g["flag"] or "no flag",
          bool(g["flag"]))
    L2 = copy.deepcopy(s.L)
    L2.groups["b002"]["flag"] = ""
    weak = [dict(r, score=-10.10) if r["config"] == "9-4-4-8-0" else r for r in s.rows]
    L2.update_flags(dict(h4, verdict=""), None, [r for r in weak if r["id"].startswith("b004")], weak, CAL25["tau"])
    c.add("A no flag below τ (breaker better by 0.10)", "no flag", L2.groups["b002"]["flag"] or "no flag",
          not L2.groups["b002"]["flag"])
    L2.update_flags(dict(h4, verdict="refuted", hyp_id="b099"), "b002", [], weak, CAL25["tau"])
    c.add("A flag: refuted test of the rule", "b002 flagged", L2.groups["b002"]["flag"] or "no flag",
          "b099 gave refuted" in L2.groups["b002"]["flag"])

    errs, _ = s._check_reply(reply("y0=9", "x0=5; x1=3", parent="b004", exceptions=exc))
    c.add("A missing flag decision", "rejected", errs[0] if errs else "accepted", has(errs, "rule b002 is flagged"))
    dec = [dict(rule_id="b002", decision="demote", reason="b004 beats it by more than τ")]
    errs, plan = s._check_reply(reply("y0=9", "x0=5; x1=3", decisions=dec, parent="b004"))
    c.add("A flag decision given", "accepted", f"errors {errs}", not errs)
    h5 = simulate(s, "b005", plan, {}, default=-9.00)
    c.add("A demotion", "b002 short, flag cleared", f"b002 {g['level']}, flag {g['flag'] or 'none'}",
          g["level"] == "short" and not g["flag"])
    g4 = s.L.groups["b004"]
    c.add("A no promotion again without a new supported test; b005 in group b004",
          "b002 short; b004 refuted 1/1/0, evidence [b004, b005]",
          f"b002 {g['level']}; b004 {g4['status']} {g4['n_support']}/{g4['n_refute']}/{g4['n_inconclusive']}, "
          f"{g4['evidence']}",
          g["level"] == "short" and h5["verdict"] == "refuted" and g4["evidence"] == ["b004", "b005"]
          and (g4["n_support"], g4["n_refute"]) == (1, 1))

    Lx = learnings.Learnings(25)
    for k in range(7):
        Lx.record_test(Lx.assign(f"y0={k}", "lower", f"b{10 + k:03d}"), f"b{10 + k:03d}", "inconclusive")
    recent = [x["group_id"] for x in Lx.recent_short()]
    c.add("A short-term list: the 5 latest tested groups", "[b012 … b016]", str(recent),
          recent == ["b012", "b013", "b014", "b015", "b016"])

    tree = {"b001": dict(parent_id=""), "b002": dict(parent_id="b001"), "b003": dict(parent_id="b002"),
            "b004": dict(parent_id="")}
    p1, p2 = hypothesis.path_to(tree, "b003", "b001"), hypothesis.path_to(tree, "b004", "b001")
    c.add("A back step closes the branch up to the parent", "[b003, b002]; [] if not an ancestor", f"{p1}; {p2}",
          (p1, p2) == (["b003", "b002"], []))

    cs = {n: noise.expected_max(n) for n in C_TABLE}
    d = noise.expected_max(20) - noise.expected_max(5)
    c.add("A c(n) against design Y14", "c(2)…c(20) within 0.005; c(20) − c(5) = 0.71 ± 0.01",
          ", ".join(f"c({n}) = {v:.4f}" for n, v in cs.items()) + f"; diff {d:.4f}",
          all(abs(cs[n] - C_TABLE[n]) < 0.005 for n in C_TABLE) and abs(d - 0.71) < 0.01)

    bests, k, ks = [-9.40, -9.70, -9.75, -9.80, -9.85], 0, []
    for b0, b1 in zip([None] + bests[:-1], bests):
        k = harness.no_gain(k, b0, b1, CAL25["eps_eff"])
        ks.append(k)
    c.add("A no-gain counter (ε_eff = 0.1973)", "[0, 0, 1, 2, 3]", str(ks), ks == [0, 0, 1, 2, 3])

    st = s._state(3600.0, ["- mark"])
    pos = [st.find(h) for h in STATE_HEADERS]
    c.add("A state blocks in the order of X8", "increasing positions", str(pos),
          all(p >= 0 for p in pos) and pos == sorted(pos))
    (out / "state_part_a.txt").write_text(st + "\n")


def replies_b():
    """Fixed replies of the fake planner of part B."""
    h2 = dict(parent_id="b001", step="child", region="y0=6", context="x0=1; x1=1", prediction="lower", ref_id="",
              rationale="smoke", exceptions=[])
    h3 = dict(h2, parent_id="b002", context="x0=2; x1=2")
    return [dict(flag_decisions=[], hypothesis=dict(h2, region="y0>=0", context="")),
            dict(flag_decisions=[], hypothesis=h2),
            dict(flag_decisions=[], hypothesis=dict(h3, context="x0<2")),
            dict(flag_decisions=[], hypothesis=h3)]


def part_b(c, out):
    repo, outputs = out / "repo", out / "outputs"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "smoke"), ("config", "user.email", "smoke@localhost")):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    cal = dict(id="synthetic", file="", R=2, sigma_R=0.05, tau=float(noise.tau(0.05)), eps=100.0, eps_eff=100.0,
               N=10, t_start=0.1, maxiter=50, kill_time=60.0)
    sid = "n10_learn_smoke"
    s = harness.LearnSession(session_id=sid, n=10, m=20, maxiter=50, kill_time=None, budget_s=math.inf,
                             planner=planner.FakeLearnPlanner(replies_b()), noise_cal=cal, b_max=4, k_stop=2,
                             n_promote=2, n_confirm=2, root=repo, out_root=outputs, commit_mode="batch",
                             allow_dirty=True, banks=P.load_banks())
    t0 = time.perf_counter()
    meta = s.run()
    c.info("B session wall time", f"{time.perf_counter() - t0:.1f} s (trials {meta['wall_trials_s']:.1f} s, "
                                  f"confirmation {meta['wall_confirm_s']:.1f} s)")
    lg = log.Log(repo / "sessions" / sid)
    rows, starts, hyps, calls = lg.results(), lg.starts(), lg.hypotheses(), lg.calls()
    c.add("B stop rule", "no gain > eps_eff in 2 batches", meta["stop"], meta["stop"] == "no gain > eps_eff in 2 batches")
    header = lg.results_path.read_text().splitlines()[0].split(",")
    c.add("B results.csv columns", "v1 columns", ",".join(header), tuple(header) == log.RESULT_COLUMNS)
    c.add("B trials", "3 + 1 + 1 = 5", str(len(rows)), len(rows) == 5)
    by_id = {}
    for x in starts:
        by_id.setdefault(x["id"], []).append(x)
    ok = all(sorted(x["seed"] for x in by_id.get(r["id"], [])) == [0, 1]
             and sum(x["selected"] for x in by_id[r["id"]]) == (r["status"] != "crash") for r in rows)
    c.add("B starts.csv", "2 starts (seeds 0, 1) per trial, one selected", f"{len(starts)} rows", ok)
    hyp_header = lg.hyp_path.read_text().splitlines()[0].split(",")
    reuse2 = next(h["reused_ids"] for h in hyps if h["hyp_id"] == "b002")
    want2 = next(r["id"] for r in rows if r["config"] == "6-1-1-1-1")
    c.add("B hypotheses.csv", "region and context columns; b001 no verdict; b002 reuses 6-1-1-1-1",
          f"{[(h['hyp_id'], h['region'], h['context']) for h in hyps]}, b001 verdict {hyps[0]['verdict'] or 'none'}, "
          f"reused {reuse2}",
          tuple(hyp_header) == log.HYP_COLUMNS and len(hyps) == 3 and not hyps[0]["verdict"] and reuse2 == want2
          and all(h["verdict"] in hypothesis.VERDICTS for h in hyps[1:]))
    call_header = lg.calls_path.read_text().splitlines()[0].split(",")
    seq = [(x["call_id"], int(x["retry"])) for x in calls]
    c.add("B calls.csv with the retry column", "[(c002, 0), (c002, 1), (c003, 0), (c003, 1)]", str(seq),
          tuple(call_header) == log.CALL_COLUMNS and seq == [("c002", 0), ("c002", 1), ("c003", 0), ("c003", 1)])
    r1 = json.loads((outputs / sid / "calls" / "c002_r1.json").read_text())
    r3 = json.loads((outputs / sid / "calls" / "c003_r1.json").read_text())
    c.add("B retry messages", "error lists of the rejected replies sent back",
          f"{len(r1['messages'])} messages in c002 retry 1",
          len(r1["messages"]) == 3 and "more than B_max = 4" in r1["messages"][-1]["content"]
          and "context 'x0<2'" in r3["messages"][-1]["content"])
    st = json.loads((outputs / sid / "calls" / "c003_r0.json").read_text())["messages"][0]["content"]
    pos = [st.find(h) for h in STATE_HEADERS if h != "## Your previous call"]
    c.add("B state blocks in the order of X8", "increasing positions", str(pos),
          all(p >= 0 for p in pos) and pos == sorted(pos))
    gr = {x["group_id"]: x for x in lg.learnings()}
    g = gr.get("b002", {})
    c.add("B learnings.csv", "one group b002: region y0=6, evidence b002 b003",
          f"{list(gr)}, {g.get('region')}, {g.get('evidence')}",
          list(gr) == ["b002"] and g.get("region") == "y0=6" and g.get("evidence") == "b002 b003")
    conf, cst = lg.confirm(), lg.confirm_starts()
    c.add("B confirmation", "2 rows; 4 starts with seeds 2, 3", f"{len(conf)} rows, {len(cst)} starts",
          len(conf) == 2 and sorted({x["seed"] for x in cst}) == [2, 3] and len(cst) == 4)
    n_commits = int(subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    c.add("B commits", "3 batches + end = 4", str(n_commits), n_commits == 4)
    z = np.load(outputs / sid / rows[0]["id"] / "trial.npz")
    c.add("B trial.npz", "X_all (2, 10, 3), X (10, 3)", f"{z['X_all'].shape}, {z['X'].shape}",
          z["X_all"].shape == (2, 10, 3) and z["X"].shape == (10, 3))

    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "scripts" / "plot_learn_session.py"), sid,
                        "--root", str(repo), "--out-root", str(outputs)], capture_output=True, text=True)
    plots = sorted(p.name for p in (outputs / sid / "plots").glob("W*.png"))
    c.add("B plot_learn_session.py", "W7–W12 and summary.txt", f"{plots} in {time.perf_counter() - t0:.1f} s"
          + ("" if r.returncode == 0 else r.stderr[-600:]),
          r.returncode == 0 and len(plots) == 6 and (outputs / sid / "plots" / "summary.txt").exists())
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "scripts" / "plot_session.py"), "--llm", sid,
                        "--root", str(repo), "--out-root", str(outputs)], capture_output=True, text=True)
    c.add("B plot_session.py on the learn session", "exit 0", f"exit {r.returncode} in {time.perf_counter() - t0:.1f} s"
          + ("" if r.returncode == 0 else r.stderr[-600:]), r.returncode == 0)


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_learn"))
    c, t_all = Checks(), time.perf_counter()
    t0 = time.perf_counter()
    part_a(c, out)
    c.info("A wall time", f"{time.perf_counter() - t0:.1f} s")
    part_b(c, out)
    c.info("total wall time", f"{time.perf_counter() - t_all:.1f} s")
    c.rows.append(("RESULT", "all PASS", "", "PASS" if c.ok else "FAIL"))
    with open(out / "checks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("case", "expected", "result", "pass"))
        w.writerows(c.rows)
    print(f"RESULT {'PASS' if c.ok else 'FAIL'}\n-> {out}")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
