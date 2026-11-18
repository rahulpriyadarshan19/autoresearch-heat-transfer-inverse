"""Plots of case2_ar_smoke_multistart: final Φ_train of both paths (V8a), best-of-R statistics (V8b), time per
start-iteration against batch size (V9), and the loss of one start of each sensor_layout on both paths.

Run:  venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_multistart.py outputs/case2_ar_smoke_multistart_<stamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoresearch import plotting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    d = ap.parse_args().run_dir
    plotting.set_style()
    st = pd.read_csv(d / "starts.csv")
    g, s = (st[st.path == p].reset_index(drop=True) for p in ("gpu", "scipy"))
    cfgs = list(dict.fromkeys(g.config))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cfg in enumerate(cfgs):
        sel = g.config == cfg
        ax.scatter(s.phi_train[sel], g.phi_train[sel], s=36, color=f"C{i}", alpha=0.8, label=cfg)
    lim = [min(g.phi_train.min(), s.phi_train.min()), max(g.phi_train.max(), s.phi_train.max())]
    ax.plot(lim, lim, color="0.4", lw=1, ls="--", label="identity")
    ax.set_xlabel(r"$\Phi_{\mathrm{train}}$, SciPy L-BFGS-B")
    ax.set_ylabel(r"$\Phi_{\mathrm{train}}$, batched GPU L-BFGS")
    ax.legend(loc="upper left")
    plotting.save(fig, d / "phi_train_gpu_vs_scipy.png")

    bs = pd.read_csv(d / "bootstrap.csv")
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6.2 * len(cfgs), 5.5))
    for ax, cfg in zip(np.atleast_1d(axes), cfgs):
        b = bs[bs.config == cfg]
        ax.fill_between(b.R, b.mean_best_scipy - b.tau, b.mean_best_scipy + b.tau, color="0.85",
                        label=r"SciPy $\pm\,\tau(R)$")
        ax.plot(b.R, b.mean_best_scipy, "o-", color="C0", label="SciPy L-BFGS-B")
        ax.plot(b.R, b.mean_best_gpu, "s--", color="C3", label="batched GPU L-BFGS")
        ax.set_xscale("log", base=2)
        ax.set_xticks(b.R, [str(r) for r in b.R])
        ax.set_xlabel(r"starts $R$")
        ax.set_title(cfg)
    np.atleast_1d(axes)[0].set_ylabel(r"mean $\Phi_{\mathrm{val}}$ of the best of $R$")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    plotting.save(fig, d / "best_of_R.png")

    t = pd.read_csv(d / "timing.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t.batch, 1e3 * t.s_per_start_iter, "o-", color="C0", label="batched GPU, per start")
    ax.axhline(1e3 * t.scipy_s_per_iter.iloc[0], color="C3", ls="--", label="SciPy, one start")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15e3 * t.s_per_start_iter.max())
    ax.set_xlabel(r"batch size $b$")
    ax.set_ylabel(r"time per start-iteration (ms)")
    ax.legend(loc="lower right")
    plotting.save(fig, d / "timing.png")

    h = np.load(d / "hist.npz")
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cfg in enumerate(cfgs):
        j = int(np.flatnonzero(h["config"] == cfg)[0])
        hg, hs = h["hist_gpu"][j, 1:], h["hist_scipy"][j]
        ref = np.nanmin(np.concatenate([hg, hs]))
        ax.semilogy(np.arange(1, len(hg) + 1), hg - ref + 1e-12, color=f"C{i}", label=f"{cfg}, GPU")
        ax.semilogy(np.arange(1, len(hs) + 1), hs - ref + 1e-12, color=f"C{i}", ls="--", label=f"{cfg}, SciPy")
    ax.set_xscale("log")
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$\mathcal{L}(k) - \mathcal{L}_{\min}$")
    ax.set_title(plotting.tex(f"seed {int(h['seed'][0])} of each sensor_layout"))
    ax.legend(loc="lower left", fontsize=12, ncol=2)
    plotting.save(fig, d / "loss_traces.png")
    print(f"-> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
