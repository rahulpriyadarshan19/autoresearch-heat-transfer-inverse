"""Plots of case2_ar_smoke_trial: start and final positions of both trials, and the loss of the rerun.

Run:  venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_trial.py outputs/case2_ar_smoke_trial_<stamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoresearch import plotting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    d = ap.parse_args().run_dir
    plotting.set_style()
    for name, title, start in (("trial", "6-6-6-6-6, maxiter 50", "start (Halton)"),
                               ("rerun", "dopt_x1z1_5 rerun, maxiter 1000", "start (§6.1 design)")):
        z = np.load(d / f"{name}.npz")
        fig, axes = plotting.face_figure(title)
        plotting.plot_design(axes, z["X_start"], z["X"], z["faces"])
        plotting.design_legend(fig, start_label=start)
        plotting.save(fig, d / f"positions_{name}.png")
    z = np.load(d / "rerun.npz")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(np.arange(1, len(z["loss"]) + 1), z["loss"] - z["loss"].min() + 1e-12)
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$\mathcal{L}(k) - \mathcal{L}_{\min}$")
    ax.set_title(plotting.tex("rerun of dopt_x1z1_5: training loss"))
    plotting.save(fig, d / "loss_rerun.png")
    print(f"-> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
