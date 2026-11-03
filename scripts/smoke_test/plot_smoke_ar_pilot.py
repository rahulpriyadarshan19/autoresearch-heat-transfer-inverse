"""Plots V1–V4 of case2_ar_smoke_pilot; V5 (determinism) is in pilot_summary.txt.

Run:  venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_pilot.py outputs/case2_ar_smoke_pilot_<stamp>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoresearch import log, plotting

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_ar_pilot as pilot  # noqa: E402

PHI = r"$\Phi_{\mathrm{val}}$"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    d = ap.parse_args().run_dir
    plotting.set_style()
    A = pilot.read_trials(d / "A")
    distinct = A[:5]

    fig, ax = plt.subplots(figsize=(11, 7))
    for r in distinct:
        loss = np.load(d / "A" / f"{r['k']:02d}_{r['config']}.npz")["loss"]
        gap = loss - loss[-1]
        ax.semilogy(np.arange(1, len(loss) + 1)[gap > 0], gap[gap > 0], label=r["config"])
    ax.axvline(pilot.SNAPSHOT, color="k", ls="--", lw=1.2, label=r"$k = 1000$")
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{train}}(k) - \mathcal{L}_{\mathrm{train}}(\mathrm{final})$")
    ax.set_title("V1: L-BFGS-B convergence, maxiter 5000")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plotting.save(fig, d / "V1_convergence.png")

    fig, ax = plt.subplots(figsize=(9, 8))
    pts = [(r["score"], r["score_at_snapshot"], r["config"]) for r in A if r["score_at_snapshot"] is not None]
    for i, (x, y, cfg) in enumerate(pts):
        ax.scatter(x, y, s=70, color=f"C{i}", label=cfg, zorder=3)
    if pts:
        lo = min(min(p[0], p[1]) for p in pts)
        hi = max(max(p[0], p[1]) for p in pts)
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, label=r"$y = x$")
    ax.set_xlabel(f"{PHI} at convergence")
    ax.set_ylabel(f"{PHI} at iteration 1000")
    ax.set_title("V2: score at 1000 iterations against the final score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plotting.save(fig, d / "V2_score_at_1000.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    cfgs = pilot.LAYOUTS_B
    by_cfg = {r["config"]: r for r in distinct}
    x = np.arange(len(cfgs))
    per_it = lambda r: 1e3 * r["wall_optimise"] / r["nit"]  # noqa: E731
    series = [("GPU 0 alone", [per_it(by_cfg[c]) for c in cfgs])]
    for g in (0, 1):
        rows = {r["config"]: r for r in pilot.read_trials(d / f"B_gpu{g}")}
        series.append((f"GPU {g}, both GPUs busy", [per_it(rows[c]) for c in cfgs]))
    for i, (lab, vals) in enumerate(series):
        ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=lab)
    ax.set_xticks(x, cfgs)
    ax.set_ylabel("time per L-BFGS-B iteration (ms)")
    ax.set_title("V3: CPU slowdown with two processes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plotting.save(fig, d / "V3_slowdown.png")

    calls = log.Log(d / "C").calls()
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, e in enumerate(pilot.EFFORTS):
        cs = [c for c in calls if c["call_id"].startswith(e)]
        w = [float(c["wall_time"]) for c in cs]
        ax.bar(np.arange(len(w)) + i * (len(w) + 1), w, color=f"C{i}",
               label=f"effort {e}: mean {np.mean(w):.1f} s, {np.mean([float(c['cost']) for c in cs]):.3f} USD/call")
    ax.set_xticks([])
    ax.set_xlabel("planner call")
    ax.set_ylabel(r"$t_{\mathrm{LLM}}$ (s)")
    ax.set_title(plotting.tex("V4: planner call time (claude-opus-5-5, B = 5)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    plotting.save(fig, d / "V4_llm_time.png")
    print(f"-> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
