"""Plot P1 of case2_ar_posterior: prior and bank-averaged Laplace posterior contours of saved session designs.

Run:  venv_ar/bin/python scripts/plot_posterior.py outputs/case2_ar_posterior_<stamp>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoresearch import plotting

STAGE_TEXT = {"start": "Halton start", "final": "after L-BFGS-B"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    d = ap.parse_args().run_dir
    plotting.set_style()
    with open(d / "posterior.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    z = np.load(d / "posterior.npz")
    t0 = z["theta0"]
    designs = [r for r in rows if r["stage"] != "prior"]
    finals = [r for r in designs if r["stage"] == "final"]

    fig = plt.figure(figsize=(10, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.28)
    ax_all, ax_fin = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    prior = next(r for r in rows if r["stage"] == "prior")
    p = t0 + z["outline_prior"]
    ax_all.plot(p[:, 0], p[:, 1], "--", color="0.35", lw=2.5,
                label=rf"prior: $\log\det\Sigma_0 = {float(prior['phi_val']):.3f}$")
    for i, r in enumerate(designs):
        q = t0 + z[f"outline_{r['label']}"]
        label = plotting.tex(f"{r['trial']}, {r['config']}, {STAGE_TEXT[r['stage']]}: "
                             rf"$\Phi_{{\mathrm{{val}}}} = {float(r['phi_val']):.3f}$")
        ax_all.plot(q[:, 0], q[:, 1], color=f"C{i}", lw=2.5, label=label)
        if r in finals:
            ax_fin.plot(q[:, 0], q[:, 1], color=f"C{i}", lw=2.5)
    for ax in (ax_all, ax_fin):
        ax.plot(*t0, "+", color="k", ms=14, mew=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"$\theta_1 = \log k_{xx}$")
        ax.set_ylabel(r"$\theta_2 = \log k_{yy}$")
    ax_all.set_title("(a) prior and posteriors")
    ax_fin.set_title("(b) posteriors after L-BFGS-B")
    fig.suptitle(plotting.tex(r"95% contours of the log-Euclidean mean over 128 held-out $\theta_s$, "
                              r"centred at $\theta_0$"), y=0.935, fontsize=16)
    fig.legend(*ax_all.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=1,
               frameon=False)
    plotting.save(fig, d / "plots" / "P1_posterior.png")
    print(d / "plots" / "P1_posterior.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
