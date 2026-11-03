"""Plot style and the face-net plot of a design; plots read saved data only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import problem as P  # noqa: E402

AXIS_NAME = ("x", "y", "z")
FACE_TEX = {"y0": r"$y_0$", "x0": r"$x_0$", "x1": r"$x_1$", "z0": r"$z_0$", "z1": r"$z_1$"}


def set_style() -> bool:
    """Seaborn theme with LaTeX text (serif); returns False if no TeX install renders, then mathtext is used."""
    sns.set_theme(context="talk")
    plt.rcParams.update({"font.family": "serif", "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                         "figure.titlesize": 20, "text.usetex": True})
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\Phi_{\mathrm{val}}$ test")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:  # matplotlib raises only at draw time when TeX is missing
        plt.close("all")
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})
        return False


def tex(s: str) -> str:
    """Escape LaTeX special characters outside $...$ when usetex is on."""
    if not plt.rcParams["text.usetex"]:
        return s
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("§", r"\S{}")):
            parts[i] = parts[i].replace(a, b)
    return "$".join(parts)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def face_figure(title=None):
    """Figure with one equal-aspect panel per allowed face, axes in mm -> (fig, {face: ax})."""
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.4], hspace=0.35)
    top = gs[0].subgridspec(1, 3, width_ratios=[1, 4, 1], wspace=0.35)
    bottom = gs[1].subgridspec(1, 2, wspace=0.25)
    cells = {"x0": top[0], "y0": top[1], "x1": top[2], "z0": bottom[0], "z1": bottom[1]}
    axes = {}
    for name, cell in cells.items():
        ax = fig.add_subplot(cell)
        h, v = P.IN_PLANE[P.FACE[name].axis]
        ax.set_xlim(0, 1e3 * P.L3[h])
        ax.set_ylim(0, 1e3 * P.L3[v])
        lo, hi = 1e3 * P.M_LO, 1e3 * (P.L3 - P.M_HI)
        ax.plot([lo[h], hi[h], hi[h], lo[h], lo[h]], [lo[v], lo[v], hi[v], hi[v], lo[v]], ls="--", lw=0.8,
                color="0.5")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(rf"${AXIS_NAME[h]}$ (mm)")
        ax.set_ylabel(rf"${AXIS_NAME[v]}$ (mm)")
        ax.set_title(FACE_TEX[name])
        axes[name] = ax
    if title:
        fig.suptitle(tex(title))
    return fig, axes


def plot_design(axes, X_start, X, faces, color="C0"):
    """Start (hollow), final (filled) and a thin grey line from start to final for each sensor."""
    X_start, X, faces = 1e3 * np.asarray(X_start), 1e3 * np.asarray(X), np.asarray(faces)
    for name, ax in axes.items():
        sel = faces == name
        if not sel.any():
            continue
        h, v = P.IN_PLANE[P.FACE[name].axis]
        for a, b in zip(X_start[sel], X[sel]):
            ax.plot([a[h], b[h]], [a[v], b[v]], color="0.55", lw=0.8, zorder=1)
        ax.scatter(X_start[sel, h], X_start[sel, v], s=40, facecolors="none", edgecolors=color, lw=1.4, zorder=2)
        ax.scatter(X[sel, h], X[sel, v], s=40, color=color, zorder=3)


def design_legend(fig, color="C0", start_label="start (Halton)"):
    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec=color, ms=8, label=tex(start_label)),
               Line2D([], [], ls="none", marker="o", color=color, ms=8, label="final (L-BFGS-B)"),
               Line2D([], [], color="0.55", lw=0.8, label="start to final")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
