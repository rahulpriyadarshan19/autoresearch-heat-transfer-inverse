"""ID case2_ar_smoke_pilot: pilot at N = 30 that sets maxiter, kill time, B, M, the cache TTL and mirror_reduce.

Context: scripts/run_session.py needs these values before the first LLM session and exhaustive run at N = 30.
Phases, run in order by run_pilot.sh:
    A        GPU 0 alone, maxiter 5000, design saved at iteration 1000: 5 sensor_layouts, a repeat of
             6-6-6-6-6 (determinism) and the mirror images of 20-5-0-5-0 and 0-8-7-8-7 (mirror reduction)
    B        GPU 0 and GPU 1 at the same time, 3 of the sensor_layouts on each (CPU slowdown)
    C        6 planner calls through the proxy with the state of A (B = 5): 3 at effort medium, 3 at high;
             the replies are checked, not run (t_LLM, tokens, cost)
    summary  pilot_summary.txt: measured values and the proposed settings
Outputs: outputs/case2_ar_smoke_pilot_<stamp>/{A, B_gpu0, B_gpu1, C, pilot_summary.txt}; plots V1–V4 with
plot_smoke_ar_pilot.py.

Run:  scripts/smoke_test/run_pilot.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from autoresearch import layout, log, planner, state
from heat_inverse import config

N, MAXITER, SNAPSHOT = 30, 5000, 1000
LAYOUTS_A = ("30-0-0-0-0", "6-6-6-6-6", "20-5-0-5-0", "0-8-7-8-7", "10-10-0-10-0",
             "6-6-6-6-6", "20-0-5-0-5", "0-7-8-7-8")
REPEAT = (1, 5)  # indices into LAYOUTS_A
MIRRORS = ((2, 6), (3, 7))
LAYOUTS_B = ("6-6-6-6-6", "20-5-0-5-0", "10-10-0-10-0")
B_PILOT, EFFORTS, CALLS_PER_EFFORT = 5, ("medium", "high"), 3
PROXY = "http://127.0.0.1:8765"
PRICES = dict(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0)  # claude-opus-5-5, USD per MTok
STATE_BUDGET = 20000  # tokens of state per call, for M
TRIAL_COLUMNS = ("k", "config", "nit", "converged", "message", "wall_optimise", "wall_total", "t_at_snapshot",
                 "score", "score_at_snapshot", "s1", "s2", "rho", "min_spacing_mm", "feasible")


def run_trials(d: Path, configs):
    import jax

    from autoresearch import evaluator, trial
    from autoresearch import problem as P

    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    d.mkdir(parents=True, exist_ok=True)
    B_train, B_val = P.load_banks()
    # compile once before the timed trials, so that no trial time includes the JIT compile
    Xw, fw = layout.start_design(layout.parse(configs[0], N))
    evaluator.evaluate(trial.optimise(Xw, fw, B_train, 2).X, fw, B_val)
    with open(d / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIAL_COLUMNS)
        w.writeheader()
        for k, cfg in enumerate(configs):
            t0 = time.perf_counter()
            X0, faces = layout.start_design(layout.parse(cfg, N))
            r = trial.optimise(X0, faces, B_train, MAXITER, snapshot_iter=SNAPSHOT)
            ev = evaluator.evaluate(r.X, faces, B_val)
            snap = evaluator.evaluate(r.X_snapshot, faces, B_val)["score"] if r.X_snapshot is not None else ""
            wall = time.perf_counter() - t0
            np.savez(d / f"{k:02d}_{cfg}.npz", X_start=r.X_start, X=r.X, faces=r.faces, loss=r.loss, t_iter=r.t_iter,
                     X_snapshot=r.X_snapshot if r.X_snapshot is not None else np.zeros((0, 3)))
            w.writerow(dict(k=k, config=cfg, nit=r.nit, converged=r.converged, message=r.message,
                            wall_optimise=r.wall_time, wall_total=wall,
                            t_at_snapshot=r.t_iter[SNAPSHOT - 1] if len(r.t_iter) >= SNAPSHOT else "",
                            score=ev["score"], score_at_snapshot=snap, s1=ev["s1"], s2=ev["s2"], rho=ev["rho"],
                            min_spacing_mm=ev["min_spacing_mm"], feasible=ev["feasible"]))
            fh.flush()
            print(f"{k} {cfg}: nit {r.nit}, {wall:.1f} s, score {ev['score']:.6f}", flush=True)


def read_trials(d: Path):
    with open(d / "trials.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for c in ("wall_optimise", "wall_total", "score", "s1", "s2", "rho", "min_spacing_mm"):
            r[c] = float(r[c])
        for c in ("t_at_snapshot", "score_at_snapshot"):
            r[c] = float(r[c]) if r[c] else None
        r["nit"], r["k"] = int(r["nit"]), int(r["k"])
        r["converged"], r["feasible"] = r["converged"] == "True", r["feasible"] == "True"
    return rows


def state_rows(trials):
    """The distinct sensor_layouts of phase A as batch b001 of a session log."""
    rows, seen = [], set()
    for r in trials:
        key = layout.canonical(layout.parse(r["config"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(id=f"b001-{len(rows) + 1:02d}", config=r["config"], score=r["score"], s1=r["s1"],
                         s2=r["s2"], rho=r["rho"], status="discard", description="pilot sensor_layout"))
    best = min(rows, key=lambda r: r["score"])
    best["status"] = "keep"
    return rows


def run_planner(d: Path):
    rows = state_rows(read_trials(d / "A"))
    tried = {layout.canonical(layout.parse(r["config"])) for r in rows}
    st = state.build(rows, {}, n=N, b=B_PILOT, m=50, remaining_s=36000.0,
                     untried=len(layout.enumerate_layouts(N)) - len(rows))
    out = d / "C"
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "state.txt").write_text(st)
    lg = log.Log(out)
    program = (config.PROJECT_ROOT / "program.md").read_text()
    checks = []
    for effort in EFFORTS:
        p = planner.AnthropicPlanner(program, model="claude-opus-5-5", effort=effort, base_url=PROXY, prices=PRICES)
        for i in range(CALLS_PER_EFFORT):
            cid = f"{effort}_{i + 1}"
            reply = p.propose(st, cid)
            lg.append_call(reply.call)
            reply.raw["state"] = st
            (out / "calls" / f"{cid}.json").write_text(json.dumps(reply.raw, indent=1, default=str))
            ok = 0
            for cfg, desc in reply.proposals[:B_PILOT]:
                try:
                    ok += layout.canonical(layout.parse(cfg, N)) not in tried and len(desc) <= 100
                except ValueError:
                    pass
            checks.append(dict(call_id=cid, proposals=len(reply.proposals), valid_new=ok,
                               configs=" ".join(c for c, _ in reply.proposals)))
            print(f"{cid}: {reply.call['wall_time']:.1f} s, {reply.call['output_tokens']} output tokens, "
                  f"{ok}/{len(reply.proposals)} valid new proposals", flush=True)
    with open(out / "replies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("call_id", "proposals", "valid_new", "configs"))
        w.writeheader()
        w.writerows(checks)


def summary(d: Path):
    A = read_trials(d / "A")
    B = {g: read_trials(d / f"B_gpu{g}") for g in (0, 1)}
    calls = log.Log(d / "C").calls()
    replies = list(csv.DictReader(open(d / "C" / "replies.csv")))
    st = (d / "C" / "state.txt").read_text()
    L = []

    L.append("== A: GPU 0 alone, maxiter 5000 ==")
    for r in A:
        L.append(f"{r['k']} {r['config']:>13}: nit {r['nit']:5d}  converged {r['converged']!s:5}  wall {r['wall_total']:6.1f} s"
                 f"  t(1000) {r['t_at_snapshot'] or float('nan'):6.1f} s  score {r['score']:.6f}  score(1000) "
                 f"{r['score_at_snapshot'] if r['score_at_snapshot'] is not None else float('nan'):.6f}  "
                 f"feasible {r['feasible']}")
    distinct = [A[i] for i in range(5)]
    scores = np.sort([r["score"] for r in distinct])
    gap = float(np.diff(scores).min())
    d1000 = [abs(r["score_at_snapshot"] - r["score"]) for r in A if r["score_at_snapshot"] is not None]
    max_d1000 = max(d1000) if d1000 else 0.0
    n_conv_1000 = sum(r["nit"] < SNAPSHOT for r in A)
    L += [f"smallest score gap between the 5 sensor_layouts: {gap:.4f}",
          f"max |score(1000) - score(final)|: {max_d1000:.4f} over {len(d1000)} trials that passed 1000 iterations",
          f"trials that converged before 1000 iterations: {n_conv_1000} of {len(A)}"]

    r0, r1 = A[REPEAT[0]], A[REPEAT[1]]
    z0 = np.load(d / "A" / f"{r0['k']:02d}_{r0['config']}.npz")
    z1 = np.load(d / "A" / f"{r1['k']:02d}_{r1['config']}.npz")
    det_dx = float(np.abs(z0["X"] - z1["X"]).max())
    deterministic = r0["score"] == r1["score"] and r0["nit"] == r1["nit"] and det_dx == 0.0
    L += ["", "== determinism (repeat of 6-6-6-6-6) ==",
          f"|Δscore| {abs(r0['score'] - r1['score']):.3e}, nit {r0['nit']} vs {r1['nit']}, max |ΔX| {det_dx:.3e} m"
          f" -> deterministic: {deterministic}"]

    L += ["", "== mirror images =="]
    mirror_ok = True
    for i, j in MIRRORS:
        ds = abs(A[i]["score"] - A[j]["score"])
        mirror_ok &= ds <= 1e-9 and A[i]["nit"] == A[j]["nit"]
        L.append(f"{A[i]['config']} vs {A[j]['config']}: |Δscore| {ds:.3e}, nit {A[i]['nit']} vs {A[j]['nit']}")
    L.append(f"-> mirror_reduce: {mirror_ok} (rule: |Δscore| <= 1e-9 and equal nit)")

    L += ["", "== B: both GPUs at the same time =="]
    by_cfg = {r["config"]: r for r in A[:5]}
    slow = []
    for g in (0, 1):
        for r in B[g]:
            a = by_cfg[r["config"]]
            f = (r["wall_optimise"] / r["nit"]) / (a["wall_optimise"] / a["nit"])
            slow.append(f)
            L.append(f"GPU {g} {r['config']:>13}: wall {r['wall_total']:6.1f} s vs {a['wall_total']:6.1f} s alone, "
                     f"time per iteration x{f:.2f}, |Δscore| vs A {abs(r['score'] - a['score']):.3e}")
    L.append(f"-> slowdown per iteration with both GPUs busy: mean x{np.mean(slow):.2f}, max x{np.max(slow):.2f}")

    L += ["", "== C: planner calls (B = 5) =="]
    t_llm = {}
    for e in EFFORTS:
        cs = [c for c in calls if c["call_id"].startswith(e)]
        rs = [r for r in replies if r["call_id"].startswith(e)]
        w = np.array([float(c["wall_time"]) for c in cs])
        t_llm[e] = float(w.mean())
        L.append(f"effort {e}: t_LLM mean {w.mean():.1f} s, max {w.max():.1f} s; output tokens mean "
                 f"{np.mean([int(c['output_tokens']) for c in cs]):.0f}; cost mean "
                 f"{np.mean([float(c['cost']) for c in cs]):.4f} USD; valid new proposals "
                 f"{sum(int(r['valid_new']) for r in rs)}/{B_PILOT * len(rs)}")
    in_tokens = np.mean([int(c["input_tokens"]) for c in calls])
    lines_rows = [ln for ln in st.splitlines() if ln.startswith("b001-")]
    tpc = in_tokens / len(st)
    tpr = tpc * np.mean([len(ln) for ln in lines_rows])
    stats_tokens = tpc * 5 * (N + 1) * len(" 10:-9.1234")
    L += [f"state: {len(st)} characters, {in_tokens:.0f} uncached input tokens -> {tpc:.3f} tokens per character",
          f"tokens per row (estimate): {tpr:.0f}; face statistics at N = 30 (estimate): {stats_tokens:.0f} tokens"]

    t_trial = float(np.mean([min(r["wall_optimise"], r["t_at_snapshot"] or r["wall_optimise"]) for r in A[:5]]))
    t_kill = 2 * max(r["t_at_snapshot"] or r["wall_optimise"] for r in A)
    L += ["", "== proposed settings (you decide) ==",
          f"maxiter 1000: max |score(1000) - score(final)| / smallest score gap = {max_d1000:.4f} / {gap:.4f} = "
          f"{max_d1000 / gap:.2f} -> {'sufficient' if max_d1000 / gap < 0.1 else 'not sufficient'} "
          f"(rule: ratio < 0.1)",
          f"t_trial at maxiter 1000 (mean of the 5): {t_trial:.1f} s; kill time = 2 x longest: {t_kill:.0f} s"]
    for e in EFFORTS:
        b = max(1, math.ceil(4 * t_llm[e] / t_trial))
        m = int((STATE_BUDGET - stats_tokens - 2 * b * tpr) // tpr)
        batch_s = t_llm[e] + b * t_trial
        L.append(f"effort {e}: B >= 4 t_LLM / t_trial = {b}; M = {m} at a state budget of {STATE_BUDGET} tokens; "
                 f"batch time {batch_s:.0f} s -> cache TTL {'5m' if batch_s < 300 else '1h'}")
    L.append(f"mirror_reduce: {mirror_ok}; deterministic: {deterministic}")
    (d / "pilot_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("A", "B", "C", "summary"), required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0, help="phase B: label of the GPU of this process")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.phase == "A":
        run_trials(args.dir / "A", LAYOUTS_A)
    elif args.phase == "B":
        run_trials(args.dir / f"B_gpu{args.gpu}", LAYOUTS_B)
    elif args.phase == "C":
        run_planner(args.dir)
    else:
        summary(args.dir)
    print(f"phase {args.phase}: {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
