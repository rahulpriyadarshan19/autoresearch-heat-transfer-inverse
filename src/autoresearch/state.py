"""State: the part of the log that the harness sends to the planner in each call."""

from __future__ import annotations

from . import layout
from . import problem as P

# printed precision of the learn-mode state
FMT = {"score": "{:.4f}", "s1": "{:.4g}", "s2": "{:.4g}", "rho": "{:+.3f}", "delta": "{:+.4f}", "tau_h": "{:.4f}"}
TRIED_PER_LINE = 10


def show(field: str, value) -> str:
    """A log value as the learn-mode state prints it; '-' if empty."""
    if value is None or value == "":
        return "-"
    return FMT[field].format(float(value)) if field in FMT else str(value)


def _batch(row_id: str) -> str:
    return row_id.split("-")[0]


def _line(r: dict, notes: dict) -> str:
    note = f" [{notes[r['id']]}]" if r["id"] in notes else ""
    if r["score"] is None:
        metrics = "no result"
    else:
        metrics = f"score {r['score']:.5f}  s1 {r['s1']:.4g}  s2 {r['s2']:.4g}  rho {r['rho']:+.3f}"
    return f"{r['id']}  {r['config']}  {metrics}  {r['status']}{note}  | {r['description']}"


def scored(rows, notes):
    """Rows with a valid score: not crashed and not infeasible."""
    return [r for r in rows if r["score"] is not None and notes.get(r["id"]) != "infeasible"]


def face_statistics(rows, notes) -> list[str]:
    """For each face: the best score among the tried sensor_layouts with each value of n_f."""
    out = []
    for k, name in enumerate(P.FACES):
        best = {}
        for r in scored(rows, notes):
            n_f = layout.parse(r["config"])[k]
            best[n_f] = min(best.get(n_f, r["score"]), r["score"])
        out.append(f"{name}: " + "  ".join(f"{n}:{best[n]:.4f}" for n in sorted(best)) if best else f"{name}: -")
    return out


def build(rows, notes, *, n, b, m, remaining_s, untried, marks=()) -> str:
    """State text: facts, best row, best M rows, last two batches, face statistics, marks of the last reply."""
    good = sorted(scored(rows, notes), key=lambda r: r["score"])
    batches = sorted({_batch(r["id"]) for r in rows})[-2:]
    best_ids = {r["id"] for r in good[:m]}
    recent = [r for r in rows if _batch(r["id"]) in batches and r["id"] not in best_ids]
    lines = [
        f"N = {n} sensors. Propose B = {b} new sensor_layouts.",
        f"Trials so far: {len(rows)}. Untried sensor_layouts: {untried}. Remaining time: {remaining_s / 3600:.2f} h.",
        "",
        "Best so far: " + (_line(good[0], notes) if good else "none"),
        "",
        f"Best {min(m, len(good))} trials (id, config y0-x0-x1-z0-z1, score, s1, s2, rho, status | description):",
        *(_line(r, notes) for r in good[:m]),
        "",
        f"Other trials of the last {len(batches)} batches:",
        *(_line(r, notes) for r in recent),
        "",
        "Best score for each n_f on each face (n_f:score):",
        *face_statistics(rows, notes),
    ]
    if marks:
        lines += ["", "Your previous reply:", *marks]
    return "\n".join(lines)


# ---------------------------------------------------------------------- learn mode


def _learn_line(r: dict, notes: dict) -> str:
    note = f" [{notes[r['id']]}]" if r["id"] in notes else ""
    metrics = "no result" if r["score"] is None else "  ".join(
        f"{k} {show(k, r[k])}" for k in ("score", "s1", "s2", "rho"))
    return f"{r['id']}  {r['config']}  {metrics}  {r['status']}{note}"


def group_lines(g: dict) -> list[str]:
    """A group of the short-term list or a rule."""
    out = [f"{g['group_id']}: region {g['region']}, predicted {g['prediction']} [{g['status']}; tests "
           f"supported/refuted/inconclusive {g['n_support']}/{g['n_refute']}/{g['n_inconclusive']}"
           + (f"; evidence {' '.join(g['evidence'])}" if g["evidence"] else "")
           + (f"; exceptions {' '.join(g['exceptions'])}" if g["exceptions"] else "") + "]"]
    if g["flag"]:
        out.append(f"    FLAGGED: {g['flag']}")
    if g["last_reason"]:
        out.append(f"    last change: {g['last_reason']}")
    return out


def hypothesis_line(h: dict, by_id: dict) -> str:
    where = f"region {h['region']}; context {h['context'] or 'none'}"
    if not h["ref_id"]:
        head = f"{h['hyp_id']} (screening, {h['step']}): {where}"
        test = "no verdict"
    else:
        ref = by_id.get(h["ref_id"])
        head = (f"{h['hyp_id']} ({h['step']}" + (f" of {h['parent_id']}" if h["parent_id"] else "")
                + f"): {where} | predicted {h['prediction']} than {h['ref_id']}"
                + (f" (score {show('score', ref['score'])})" if ref else ""))
        test = (f"Δ {show('delta', h['delta'])}, τ_h {show('tau_h', h['tau_h'])} (n_h {h['n_cmp']}, n_ref "
                f"{h['n_ref']}) -> {h['verdict'] or 'none'}")
    best = by_id.get(h["best_id"])
    best_s = f"best {h['best_id']} {best['config']} score {show('score', best['score'])}" if best else "no valid member"
    reused = f", reused {h['reused_ids']}" if h["reused_ids"] else ""
    exc = f" | exceptions: {h['exceptions']}" if h["exceptions"] else ""
    rat = f" | rationale: {h['rationale'][:200]}" if h["rationale"] else ""
    return f"{head} | S(h) {h['n_set']}, run {h['B_h']}{reused} | {best_s} | {test}{exc}{rat}"


def build_learn(rows, notes, hyps, rules, short, n_short, report, *, n, r, sigma, tau, b_max, n_promote, m,
                remaining_s, untried, total, marks=()) -> str:
    """Learn-mode state, blocks in the order of design X8: session constants, long-term list, tried sensor_layouts,
    open hypotheses, short-term list (the latest of n_short groups), trials, last batch, request."""
    by_id = {x["id"]: x for x in rows}
    good = sorted(scored(rows, notes), key=lambda x: x["score"])
    shown_h = [h for h in hyps if h["branch"] == "open"]
    last = _batch(rows[-1]["id"]) if rows else None
    blocks = [("best", good[:m]), ("last batch", [x for x in rows if _batch(x["id"]) == last])]
    seen, trial_lines = set(), []
    for name, group in blocks:
        new = [x for x in group if x["id"] not in seen]
        seen |= {x["id"] for x in new}
        trial_lines += [f"{name}:", *(_learn_line(x, notes) for x in new)] if new else []
    tried = [layout.fmt(u) for u in sorted({layout.canonical(layout.parse(x["config"])) for x in rows})]
    lines = [
        f"Session: N = {n}; R = {r} starts per sensor_layout; σ_R = {sigma:.4f}; τ = {tau:.4f}; B_max = {b_max}; "
        f"n_promote = {n_promote}.",
        "",
        f"## Long-term list (rules, {len(rules)})",
        *([s for g in rules for s in group_lines(g)] or ["none"]),
        "",
        f"## Tried sensor_layouts ({len(tried)} of {total}, canonical y0-x0-x1-z0-z1)",
        *([" ".join(tried[i:i + TRIED_PER_LINE]) for i in range(0, len(tried), TRIED_PER_LINE)] or ["none"]),
        "",
        f"## Open hypotheses ({len(shown_h)})",
        *([hypothesis_line(h, by_id) for h in shown_h] or ["none"]),
        "",
        f"## Short-term list (the {len(short)} latest tested of {n_short} groups that are not rules)",
        *([s for g in short for s in group_lines(g)] or ["none"]),
        "",
        "## Trials (id, config, score, s1, s2, rho, status [note]): best, last batch",
        *(trial_lines or ["none"]),
        "",
        "## Last batch",
        *(report or ["none"]),
    ]
    if marks:
        lines += ["", "## Your previous call", *marks]
    lines += ["", "## Request",
              f"N = {n}. Remaining time: {remaining_s / 3600:.2f} h. B_max = {b_max}. Untried sensor_layouts: "
              f"{untried}. Reply with flag_decisions and one hypothesis."]
    return "\n".join(lines)
