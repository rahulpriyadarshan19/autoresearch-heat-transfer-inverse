"""Plots W1–W6 of one N: an LLM session against the exhaustive run, from their saved logs.

Smoke test: smoke_test/smoke_ar_harness.py runs this script on a fake-planner session and an exhaustive run.

Run:  venv_ar/bin/python scripts/plot_session.py --llm <session_id> --exh <session_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoresearch import layout, plotting
from autoresearch import problem as P
from autoresearch.log import Log
from heat_inverse import config

PHI = r"$\Phi_{\mathrm{val}}$"
STYLE = {"llm": dict(color="C0", label="LLM session"), "exh": dict(color="C1", label="exhaustive (random order)")}
MODE_LABEL = {"llm": "LLM session", "fake": "random planner", "exh": "exhaustive (random order)"}


def load_run(root, out_root, sid):
    meta = json.loads((root / "sessions" / sid / "session.json").read_text())
    notes_file = out_root / sid / "notes.json"
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    lg = Log(root / "sessions" / sid)
    return dict(id=sid, rows=lg.results(), calls=lg.calls(), meta=meta, notes=notes, out=out_root / sid)


def valid(run):
    return [r for r in run["rows"] if r["score"] is not None and run["notes"].get(r["id"]) != "infeasible"]


def best_trace(run):
    """Per logged trial: running best valid row (None before the first valid one)."""
    best, out, ok = None, [], {r["id"] for r in valid(run)}
    for r in run["rows"]:
        if r["id"] in ok and (best is None or r["score"] < best["score"]):
            best = r
        out.append(best)
    return out


def hours(run):
    """Session time in h after each trial: trial wall times plus the planner call before each batch."""
    call = {c["call_id"][1:]: float(c["wall_time"]) for c in run["calls"]}
    t, seen, out = 0.0, set(), []
    for r in run["rows"]:
        b = r["id"].split("-")[0][1:]
        if b not in seen:
            seen.add(b)
            t += call.get(b, 0.0)
        t += r["wall_time"]
        out.append(t / 3600)
    return np.array(out)


def w1(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    exh_best = min(r["score"] for r in valid(runs["exh"])) if "exh" in runs else None
    for key, run in runs.items():
        tr = best_trace(run)
        y = np.array([b["score"] if b else np.nan for b in tr])
        axes[0].step(np.arange(1, len(y) + 1), y, where="post", **STYLE[key])
        axes[1].step(hours(run), y, where="post", **STYLE[key])
    for ax in axes:
        if exh_best is not None:
            ax.axhline(exh_best, color="k", ls="--", lw=1.2, label="exhaustive best")
        ax.set_xscale("log")
    axes[0].set_xlabel("trial count")
    axes[1].set_xlabel("session wall time (h)")
    axes[0].set_ylabel(f"best {PHI} so far")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [plotting.tex(s) for s in labels], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    plotting.save(fig, out / "W1_best_so_far.png")


def w2(runs, out):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    maxiter = max(run["meta"]["maxiter"] for run in runs.values())
    bins = np.linspace(0, maxiter, 41)
    for key, run in runs.items():
        nit = np.array([r["n_iter"] for r in run["rows"] if r["n_iter"] is not None])
        frac = np.mean(nit >= run["meta"]["maxiter"]) if len(nit) else np.nan
        ax.hist(nit, bins=bins, histtype="step", lw=2, density=True, color=STYLE[key]["color"],
                label=plotting.tex(f"{STYLE[key]['label']}: {100 * frac:.1f}% at maxiter"))
    ax.axvline(maxiter, color="k", ls="--", lw=1.2, label="maxiter")
    ax.set_xlabel("L-BFGS-B iterations")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plotting.save(fig, out / "W2_iterations.png")


def w3(runs, out):
    if "exh" not in runs or "llm" not in runs:
        return None
    exh = np.array([r["score"] for r in valid(runs["exh"])])
    llm = np.array([r["score"] for r in valid(runs["llm"])])
    best = llm.min()
    rank = int(np.sum(exh < best - 1e-12)) + 1
    bins = np.linspace(min(exh.min(), llm.min()), max(exh.max(), llm.max()), 50)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.hist(exh, bins=bins, density=True, alpha=0.45, color=STYLE["exh"]["color"],
            label=plotting.tex(f"exhaustive, all {len(exh)}"))
    ax.hist(llm, bins=bins, density=True, histtype="step", lw=2.2, color=STYLE["llm"]["color"],
            label=plotting.tex(f"LLM-tried, {len(llm)}"))
    ax.axvline(best, color="k", lw=1.5, label=f"LLM best: rank {rank} of {len(exh)}")
    ax.set_xlabel(PHI)
    ax.set_ylabel("density")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plotting.save(fig, out / "W3_score_histograms.png")
    return rank, len(exh)


def w4(runs, out):
    run = runs.get("exh") or runs["llm"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for k, name in enumerate(P.FACES):
        best = {}
        for r in valid(run):
            n_f = layout.parse(r["config"])[k]
            best[n_f] = min(best.get(n_f, np.inf), r["score"])
        n = np.array(sorted(best))
        ax.plot(n, [best[i] for i in n], marker="o", ms=4, label=plotting.FACE_TEX[name])
    ax.set_xlabel(r"$n_f$ (sensors on face $f$)")
    ax.set_ylabel(f"best {PHI}")
    reduced = r", mirror-reduced: $n_{x_1} \le n_{x_0}$, $n_{z_1} \le n_{z_0}$" if run["meta"]["mirror_reduce"] else ""
    ax.set_title(plotting.tex(f"{MODE_LABEL[run['meta']['mode']]}, N = {run['meta']['N']}") + reduced)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plotting.save(fig, out / "W4_face_profiles.png")


def w5(runs, out):
    for key, run in runs.items():
        b = best_trace(run)[-1]
        if b is None:
            continue
        z = np.load(run["out"] / b["id"] / "trial.npz")
        fig, axes = plotting.face_figure(f"{STYLE[key]['label']}: best {b['config']} ({b['id']}), "
                                         f"{PHI} = {b['score']:.5f}")
        plotting.plot_design(axes, z["X_start"], z["X"], z["faces"], color=STYLE[key]["color"])
        plotting.design_legend(fig, color=STYLE[key]["color"])
        plotting.save(fig, out / f"W5_positions_{key}.png")


def w6(runs, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    names = (("s1", r"$s_1$ of $\log k_{xx}$"), ("s2", r"$s_2$ of $\log k_{yy}$"), ("rho", r"$\rho$"))
    for key, run in runs.items():
        tr = best_trace(run)
        x = np.arange(1, len(tr) + 1)
        ok = {r["id"] for r in valid(run)}
        for ax, (col, _) in zip(axes, names):
            pts = [(i + 1, r[col]) for i, r in enumerate(run["rows"]) if r["id"] in ok]
            if pts:
                ax.scatter(*zip(*pts), s=6, alpha=0.15, color=STYLE[key]["color"], lw=0)
            ax.step(x, [b[col] if b else np.nan for b in tr], where="post", lw=2, **STYLE[key])
    for ax, (_, lab) in zip(axes, names):
        ax.set_ylabel(lab)
        ax.set_xscale("log")
    axes[-1].set_xlabel("trial count")
    axes[0].set_title("best design so far (line) and all trials (points), bank means")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [plotting.tex(s) for s in labels], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plotting.save(fig, out / "W6_widths.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", help="LLM (or fake-planner) session id")
    ap.add_argument("--exh", help="exhaustive session id")
    ap.add_argument("--root", type=Path, default=config.PROJECT_ROOT)
    ap.add_argument("--out-root", type=Path, default=config.OUTPUTS)
    ap.add_argument("--out", type=Path, default=None, help="default <out-root>/<first id>/plots")
    args = ap.parse_args()
    if not (args.llm or args.exh):
        ap.error("give --llm and/or --exh")
    runs = {k: load_run(args.root, args.out_root, sid) for k, sid in (("llm", args.llm), ("exh", args.exh)) if sid}
    for k, run in runs.items():
        STYLE[k]["label"] = MODE_LABEL[run["meta"]["mode"]]
    ns = {run["meta"]["N"] for run in runs.values()}
    if len(ns) != 1:
        ap.error(f"the runs have different N: {sorted(ns)}")
    out = args.out or args.out_root / (args.llm or args.exh) / "plots"
    plotting.set_style()
    w1(runs, out)
    w2(runs, out)
    rank = w3(runs, out)
    w4(runs, out)
    w5(runs, out)
    w6(runs, out)
    lines = [f"{k}: {run['id']}, {len(run['rows'])} trials, best {best_trace(run)[-1]['config']} "
             f"{best_trace(run)[-1]['score']:.6f}" for k, run in runs.items() if best_trace(run)[-1]]
    if rank:
        lines.append(f"LLM best: rank {rank[0]} of {rank[1]} exhaustive sensor_layouts")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), f"\n-> {out}", sep="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
