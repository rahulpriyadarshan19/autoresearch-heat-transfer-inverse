"""Plots of case2_ar_smoke_mirror: MS1 model symmetry, MS2 trial symmetry (old and new start rule), MS3 positions.

Run:  venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_mirror.py outputs/case2_ar_smoke_mirror_<stamp>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from autoresearch import plotting
from autoresearch import problem as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_ar_mirror as mirror  # noqa: E402

FLOOR = 1e-17  # a zero difference is drawn here on the log axis
POSITIONS = ("20-5-0-5-0", "20-0-5-0-5", "xz")  # canonical, variant, mirror


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    d = ap.parse_args().run_dir
    plotting.set_style()
    model, trials = read(d / "model.csv"), read(d / "trials.csv")

    fig, ax = plt.subplots(figsize=(10, 7))
    rng = np.random.default_rng(0)
    names = list(mirror.MIRRORS)
    for j, (col, lab) in enumerate((("d_phi_train", r"$\Phi_{\mathrm{train}}$"), ("d_phi_val", r"$\Phi_{\mathrm{val}}$"))):
        for i, name in enumerate(names):
            v = np.maximum([float(r[col]) for r in model if r["mirror"] == name], FLOOR)
            ax.scatter(i + 0.18 * (2 * j - 1) + 0.05 * rng.standard_normal(len(v)), v, s=45, color=f"C{j}",
                       alpha=0.8, label=lab if i == 0 else None, zorder=3)
    ax.axhline(mirror.TOL_PHI, color="k", ls="--", lw=1.2, label=rf"tolerance $10^{{{int(np.log10(mirror.TOL_PHI))}}}$")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), [rf"mirror ${n}$" for n in names])
    ax.set_ylabel(r"$|\Phi(mX) - \Phi(X)|$")
    ax.set_title(plotting.tex(f"MS1: model symmetry over {len(model) // len(names)} designs (0 drawn at 1e-17)"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plotting.save(fig, d / "MS1_model.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    var = [r for r in trials if r["rule"] == "new" and r["mirror"] != "none"]
    x = np.arange(len(var))
    for j, rule in enumerate(("old", "new")):
        by = {r["config"]: r for r in trials if r["rule"] == rule}
        v = np.maximum([float(by[r["config"]]["d_score"]) for r in var], FLOOR)
        lab = "Halton start of each variant (old)" if rule == "old" else "mirror of the canonical start (new)"
        ax.bar(x + 0.2 * (2 * j - 1), v, width=0.4, color=f"C{j + 2}", label=plotting.tex(lab))
    ax.axhline(mirror.TOL_TRIAL, color="k", ls="--", lw=1.2,
               label=rf"tolerance $10^{{{int(np.log10(mirror.TOL_TRIAL))}}}$")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{r['config']}\n(${r['mirror']}$ of {r['canonical']})" for r in var], fontsize=13)
    ax.set_ylabel(r"$|\Phi_{\mathrm{val}}(mL) - \Phi_{\mathrm{val}}(L)|$ after L-BFGS-B")
    ax.set_title(plotting.tex("MS2: trial result of each mirror variant against its canonical sensor_layout"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    plotting.save(fig, d / "MS2_trials.png")

    cfg_c, cfg_v, name = POSITIONS
    zc = np.load(d / "trials" / f"new_{cfg_c}.npz")
    Xm, fm = mirror.mirror_design(zc["X"], zc["faces"], mirror.MIRRORS[name])
    for rule in ("old", "new"):
        zv = np.load(d / "trials" / f"{rule}_{cfg_v}.npz")
        fig, axes = plotting.face_figure(f"MS3 ({rule} start rule): {cfg_v} against the mirror of {cfg_c}")
        plotting.plot_design(axes, zv["X_start"], zv["X"], zv["faces"], color="C0")
        for f, ax in axes.items():
            sel = fm == f
            h, v = P.IN_PLANE[P.FACE[f].axis]
            ax.scatter(1e3 * Xm[sel, h], 1e3 * Xm[sel, v], marker="x", s=70, color="C3", lw=1.6, zorder=4)
        handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec="C0", ms=8, label=f"start of {cfg_v}"),
                   Line2D([], [], ls="none", marker="o", color="C0", ms=8, label=f"final of {cfg_v}"),
                   Line2D([], [], ls="none", marker="x", color="C3", ms=9, mew=1.6,
                          label=rf"${name}$ mirror of the final of {cfg_c}")]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
        plotting.save(fig, d / f"MS3_positions_{rule}.png")
    print(f"-> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
