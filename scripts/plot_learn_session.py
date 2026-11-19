"""Plots W7–W12 of one learnings session (--mode learn) from its saved files, and summary.txt with the metrics of
design E11. The v1 plots W2, W5 and W6 of the same session: plot_session.py --llm <session_id>.

Smoke test: smoke_test/smoke_ar_learn.py (case2_ar_smoke_learn) runs this script on a fake-planner session.

Run:  venv_ar/bin/python scripts/plot_learn_session.py <session_id>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from autoresearch import plotting
from autoresearch.log import Log
from heat_inverse import config

PHI = r"$\Phi_{\mathrm{val}}$"
VERDICT_COLOR = {"supported": "C2", "refuted": "C3", "inconclusive": "C7", "": "k"}
VERDICT_LABEL = {"supported": "supported", "refuted": "refuted", "inconclusive": "inconclusive", "": "no verdict"}
STEP_MARKER = {"root": "o", "child": "^", "sibling": "s", "back": "D"}
EVENT_MARKER = {"promoted": ("P", "C1"), "demoted": ("v", "C4"), "flagged": ("X", "C3")}


def batch_no(trial_or_hyp_id: str) -> int:
    return int(trial_or_hyp_id.split("-")[0][1:])


def load(root: Path, out_root: Path, sid: str) -> dict:
    lg = Log(root / "sessions" / sid)
    meta = json.loads((root / "sessions" / sid / "session.json").read_text())
    notes_file = out_root / sid / "notes.json"
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    cal_file = Path(meta.get("calibration_file", ""))
    cal = json.loads(cal_file.read_text()) if cal_file.is_file() else None
    return dict(id=sid, meta=meta, rows=lg.results(), starts=lg.starts(), hyps=lg.hypotheses(), calls=lg.calls(),
                groups=lg.learnings(), confirm=lg.confirm(), notes=notes, cal=cal)


def learnings_history(root: Path, sid: str, groups_now) -> list[tuple[int, list[dict]]]:
    """(batch, learnings.csv rows) for each committed version; the current file only if git has none."""
    path = f"sessions/{sid}/learnings.csv"
    r = subprocess.run(["git", "-C", str(root), "log", "--reverse", "--format=%H %s", "--", path],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        h, msg = line.split(" ", 1)
        m = re.search(r"batch (\d+)", msg)
        text = subprocess.run(["git", "-C", str(root), "show", f"{h}:./{path}"], capture_output=True,
                              text=True).stdout
        out.append((int(m[1]) if m else (out[-1][0] if out else 0), list(csv.DictReader(io.StringIO(text)))))
    return out or [(0, groups_now)]


def group_events(history) -> list[tuple[int, str, str]]:
    """(batch, group id, event) from level and flag changes between committed versions of learnings.csv."""
    events, prev = [], {}
    for b, rows in history:
        for g in rows:
            old = prev.get(g["group_id"])
            if (old["level"] if old else "short", g["level"]) == ("short", "long"):
                events.append((b, g["group_id"], "promoted"))
            elif old and (old["level"], g["level"]) == ("long", "short"):
                events.append((b, g["group_id"], "demoted"))
            if g["flag"] and not (old and old["flag"]):
                events.append((b, g["group_id"], "flagged"))
        prev = {g["group_id"]: g for g in rows}
    return events


def valid(run) -> list[dict]:
    return [r for r in run["rows"] if r["score"] is not None and run["notes"].get(r["id"]) != "infeasible"]


def best_trace(run) -> np.ndarray:
    ok, best, out = {r["id"] for r in valid(run)}, np.inf, []
    for r in run["rows"]:
        if r["id"] in ok:
            best = min(best, r["score"])
        out.append(best if np.isfinite(best) else np.nan)
    return np.array(out)


def axes_values(run) -> dict:
    """Per trial: cumulative trials, starts and session hours (trial wall times plus all calls before its batch)."""
    n_starts = {}
    for s in run["starts"]:
        n_starts[s["id"]] = n_starts.get(s["id"], 0) + 1
    call_t = {}
    for c in run["calls"]:
        call_t[int(c["call_id"][1:])] = call_t.get(int(c["call_id"][1:]), 0.0) + float(c["wall_time"])
    t, done, hours = 0.0, 0, []
    for r in run["rows"]:
        b = batch_no(r["id"])
        while done < b:
            done += 1
            t += call_t.get(done, 0.0)
        t += r["wall_time"]
        hours.append(t / 3600)
    return dict(trials=np.arange(1, len(run["rows"]) + 1),
                starts=np.cumsum([n_starts.get(r["id"], 0) for r in run["rows"]]), hours=np.array(hours))


def w7(run, out):
    y, ax_v = best_trace(run), axes_values(run)
    last = {}
    for i, r in enumerate(run["rows"]):
        last[r["id"].split("-")[0]] = i
    verdict = {h["hyp_id"]: h["verdict"] for h in run["hyps"]}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, key, lab in zip(axes, ("trials", "starts", "hours"), ("trials", "starts", "session wall time (h)")):
        x = ax_v[key]
        ax.step(x, y, where="post", color="C0", lw=2)
        for hid, i in last.items():
            ax.scatter(x[i], y[i], s=90, color=VERDICT_COLOR[verdict.get(hid, "")], zorder=3, edgecolor="w")
        ax.set_xlabel(lab)
    axes[0].set_ylabel(f"best {PHI} so far")
    handles = [Line2D([], [], color="C0", lw=2, label="best so far")] + [
        Line2D([], [], ls="none", marker="o", ms=10, color=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
        for v in VERDICT_COLOR]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    plotting.save(fig, out / "W7_best_so_far.png")


def lanes(hyps) -> dict:
    """Tree layout: the first child stays in the lane of its parent; every other child and every root opens a lane."""
    lane, used, nxt = {}, set(), 0
    for h in hyps:
        p = h["parent_id"]
        if p in lane and p not in used:
            lane[h["hyp_id"]] = lane[p]
            used.add(p)
        else:
            lane[h["hyp_id"]] = nxt
            nxt += 1
    return lane


def w8(run, out):
    hyps = run["hyps"]
    lane = lanes(hyps)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(hyps) + 6), max(5, 0.5 * max(lane.values(), default=0) + 4)))
    for h in hyps:
        if h["parent_id"] in lane:
            ax.plot([batch_no(h["parent_id"]), batch_no(h["hyp_id"])], [lane[h["parent_id"]], lane[h["hyp_id"]]],
                    color="0.6", lw=1.2, zorder=1)
    for h in hyps:
        col = VERDICT_COLOR[h["verdict"]]
        ax.scatter(batch_no(h["hyp_id"]), lane[h["hyp_id"]], s=150, marker=STEP_MARKER.get(h["step"], "o"), zorder=2,
                   facecolors=col if h["branch"] == "open" else "none", edgecolors=col, lw=2)
    ax.set_xlabel("batch")
    ax.set_ylabel("branch")
    ax.invert_yaxis()
    handles = [Line2D([], [], ls="none", marker="o", ms=10, color=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
               for v in VERDICT_COLOR]
    handles += [Line2D([], [], ls="none", marker=m, ms=10, mfc="none", mec="k", label=s) for s, m in
                STEP_MARKER.items()]
    handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec="0.4", label="closed branch"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plotting.save(fig, out / "W8_hypothesis_tree.png")


def w9(run, out):
    hs = [h for h in run["hyps"] if h["verdict"]]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if hs:
        x = np.array([batch_no(h["hyp_id"]) for h in hs])
        th = np.array([h["tau_h"] for h in hs])
        ax.vlines(x, -th, th, color="0.75", lw=8, label=r"$\pm\tau_h$")
        for h, xi in zip(hs, x):
            if h["delta"] is not None:
                ax.scatter(xi, h["delta"], s=110, zorder=3, color=VERDICT_COLOR[h["verdict"]],
                           marker="v" if h["prediction"] == "lower" else "^")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("batch")
    ax.set_ylabel(rf"$\Delta$ = best member $-$ reference, {PHI}")
    handles = [Line2D([], [], color="0.75", lw=8, label=r"$\pm\tau_h$")]
    handles += [Line2D([], [], ls="none", marker="o", ms=10, color=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
                for v in ("supported", "refuted", "inconclusive")]
    handles += [Line2D([], [], ls="none", marker="v", ms=10, color="k", label="predicted lower"),
                Line2D([], [], ls="none", marker="^", ms=10, color="k", label="predicted higher")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plotting.save(fig, out / "W9_delta.png")


def w10(run, out, events):
    ids = sorted({g["group_id"] for g in run["groups"]} | {e[1] for e in events}, key=batch_no)
    row = {g: i for i, g in enumerate(ids)}
    verdict = {h["hyp_id"]: h["verdict"] for h in run["hyps"]}
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.55 * len(ids) + 2.5)))
    for g in run["groups"]:
        for hid in g["evidence"].split():
            ax.scatter(batch_no(hid), row[g["group_id"]], s=170, color=VERDICT_COLOR[verdict[hid]], zorder=2,
                       alpha=0.85)
    for b, gid, ev in events:
        m, col = EVENT_MARKER[ev]
        ax.scatter(b + 0.25, row[gid], s=120, marker=m, zorder=3, color=col)
    ax.set_yticks(range(len(ids)), [plotting.tex(i) for i in ids])
    ax.invert_yaxis()
    ax.set_xlabel("batch")
    ax.set_ylabel("group")
    handles = [Line2D([], [], ls="none", marker="o", ms=11, color=VERDICT_COLOR[v], label=f"test: {VERDICT_LABEL[v]}")
               for v in ("supported", "refuted", "inconclusive")]
    handles += [Line2D([], [], ls="none", marker=m, ms=10, color=c, label=e) for e, (m, c) in EVENT_MARKER.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plotting.save(fig, out / "W10_groups.png")


def w11(run, out):
    by = {}
    for s in run["starts"]:
        if s["valid"]:
            by.setdefault(s["id"], []).append(s["phi_val"])
    ids = [r["id"] for r in run["rows"]]
    sd = np.array([np.std(by[i], ddof=1) if len(by.get(i, [])) > 1 else np.nan for i in ids])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.scatter(np.arange(1, len(ids) + 1), sd, s=30, color="C0", label="trial: sd over its valid starts")
    if run["cal"] is not None:
        ax.axhline(run["cal"]["sigma_R"]["1"], color="C1", ls="--",
                   label=r"calibration $\sigma_1$ (one start, maximum)")
    ax.axhline(run["meta"]["sigma_R"], color="C3", ls=":",
               label=rf"calibration $\sigma_R$, $R = {run['meta']['R']}$")
    ax.set_xlabel("trial")
    ax.set_ylabel(rf"standard deviation of {PHI}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    plotting.save(fig, out / "W11_start_spread.png")


def w12(run, out):
    calls = run["calls"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    x = np.arange(1, len(calls) + 1)
    retry = np.array([int(c["retry"]) > 0 for c in calls], dtype=bool)
    for ax, key, lab in zip(axes, ("output_tokens", "wall_time", "cost"),
                            ("output tokens", r"$t_{\mathrm{LLM}}$ (s)", "cost (USD)")):
        v = np.array([float(c[key]) for c in calls])
        ax.scatter(x[~retry], v[~retry], s=45, color="C0", label="first reply")
        ax.scatter(x[retry], v[retry], s=70, marker="x", color="C3", label="corrected reply (retry)")
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("API call")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plotting.save(fig, out / "W12_calls.png")


def summary(run, out):
    """Metrics of design E11, as text."""
    meta, y, ax_v = run["meta"], best_trace(run), axes_values(run)
    lines = [f"session {run['id']}: stop '{meta.get('stop')}', {len(run['rows'])} trials, "
             f"{len(run['starts'])} starts, {len(run['hyps'])} hypotheses"]
    if np.isfinite(y).any():
        final = np.nanmin(y)
        i = int(np.argmax(y <= final + meta["eps_eff"]))
        lines += [f"best {meta.get('best_config')} ({meta.get('best_id')}): {final:.6f}",
                  f"within eps_eff = {meta['eps_eff']:.4f} of the final best after {ax_v['trials'][i]} trials, "
                  f"{ax_v['starts'][i]} starts, {ax_v['hours'][i]:.3f} h"]
    calls = run["calls"]
    t_llm = sum(float(c["wall_time"]) for c in calls)
    lines += [f"calls {len(calls)}, corrected replies (retries) {sum(int(c['retry']) > 0 for c in calls)}, "
              f"calls with nothing runnable {meta.get('n_nothing_runnable')}",
              f"cost {sum(float(c['cost']) for c in calls):.4f} USD; LLM time {t_llm:.1f} s = "
              f"{100 * t_llm / meta['elapsed_s']:.1f} % of {meta['elapsed_s']:.1f} s" if meta.get("elapsed_s") else
              "elapsed time not recorded",
              f"verdicts {meta.get('verdicts')}; rules {meta.get('n_rules')} of {meta.get('n_groups')} groups"]
    for c in run["confirm"]:
        lines.append(f"confirm {c['id']} {c['config']}: session {float(c['score_session']):.6f}, confirmed "
                     + (f"{float(c['score_confirm']):.6f}" if c["score_confirm"] else "no valid start"))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--root", type=Path, default=config.PROJECT_ROOT)
    ap.add_argument("--out-root", type=Path, default=config.OUTPUTS)
    ap.add_argument("--out", type=Path, default=None, help="default <out-root>/<session_id>/plots")
    args = ap.parse_args()
    run = load(args.root, args.out_root, args.session_id)
    out = args.out or args.out_root / args.session_id / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plotting.set_style()
    w7(run, out)
    w8(run, out)
    w9(run, out)
    w10(run, out, group_events(learnings_history(args.root, args.session_id, run["groups"])))
    w11(run, out)
    w12(run, out)
    summary(run, out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
