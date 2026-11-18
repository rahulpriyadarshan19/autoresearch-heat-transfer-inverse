"""Plots of case2_ar_smoke_noise: Φ_val of all starts per sensor_layout (V6a), τ(R) against R with ε (V6b), and the
histogram of the wall time of one start with the kill time (V7).

Run:  venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_noise.py outputs/case2_ar_smoke_noise_<stamp>
"""

from __future__ import annotations

import argparse
import json
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
    cal = json.loads((d / "calibration.json").read_text())
    cfgs = cal["configs"]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    rng = np.random.default_rng(0)
    for i, cfg in enumerate(cfgs):
        g = st[st.config == cfg]
        ok = g[g.feasible]
        x = i + rng.uniform(-0.18, 0.18, len(ok))
        ax.scatter(x, ok.phi_val, s=22, color="C0", alpha=0.6, label="valid start" if i == 0 else None)
        bad = g[~g.feasible]
        if len(bad):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(bad)), bad.phi_val, s=22, marker="x", color="C3",
                       label="spacing check failed" if "spacing" not in str(ax.get_legend_handles_labels()[1]) else None)
        if len(ok):
            best = ok.loc[ok.phi_train.idxmin()]
            ax.scatter(i, best.phi_val, s=160, marker="*", color="C1", zorder=3,
                       label=r"lowest $\Phi_{\mathrm{train}}$ of 32" if i == 0 else None)
    ax.set_xticks(range(len(cfgs)), cfgs, rotation=35, ha="right")
    ax.set_xlabel("sensor layout")
    ax.set_ylabel(r"$\Phi_{\mathrm{val}}$ of the final design")
    ax.legend(loc="upper right")
    plotting.save(fig, d / "phi_val_starts.png")

    tr = pd.read_csv(d / "tau_R.csv")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, cfg in enumerate(cfgs):
        g = tr[tr.config == cfg]
        ax.plot(g.R, g.tau_R, color="0.6", lw=1, label=r"$2\sqrt{2}\,\sigma_R$, one layout" if i == 0 else None)
    R = np.array(cal["R_grid"])
    ax.plot(R, [cal["tau_R"][str(r)] for r in R], "o-", color="C0", lw=2.5, label=r"$\tau(R)$, maximum")
    ax.axhline(cal["eps"], color="C3", ls="--", label=r"$\varepsilon = 2\ln 1.10$")
    ax.scatter([cal["R"]], [cal["tau"]], s=200, marker="*", color="C1", zorder=3, label=rf"chosen $R = {cal['R']}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(R, [str(r) for r in R])
    ax.set_xlabel(r"starts $R$")
    ax.set_ylabel(r"threshold in $\Phi_{\mathrm{val}}$")
    ax.legend(loc="upper right")
    plotting.save(fig, d / "tau_R.png")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.hist(st.wall_time, bins=40, color="C0", alpha=0.8)
    ax.axvline(cal["t_start_mean"], color="C1", ls="--", label=rf"mean $\bar t_{{\mathrm{{start}}}} = "
                                                                rf"{cal['t_start_mean']:.2f}$ s")
    ax.axvline(cal["kill_time"], color="C3", ls=":", label=rf"kill time {cal['kill_time']:.0f} s")
    ax.set_xlabel(r"wall time of one start (s)")
    ax.set_ylabel("starts")
    ax.legend(loc="upper right")
    plotting.save(fig, d / "start_time.png")
    print(f"-> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
