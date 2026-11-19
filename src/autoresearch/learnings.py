"""Learnings of the learnings run: groups of hypotheses with the same region set and prediction, the short-term list
(groups that are not rules) and the long-term list (rules); the harness-owned counters, statuses, promotions,
exceptions and flags."""

from __future__ import annotations

from . import hypothesis, layout, state

GROUP_COLUMNS = ("group_id", "level", "region", "prediction", "status", "n_support", "n_refute", "n_inconclusive",
                 "evidence", "exceptions", "flag", "last_reason")
LEVELS = ("short", "long")
MAX_SHORT, MAX_LONG = 5, 10
COUNTER = {"supported": "n_support", "refuted": "n_refute", "inconclusive": "n_inconclusive"}


def _batch_no(hyp_id: str) -> int:
    return int(hyp_id.lstrip("b"))


class Learnings:
    """All groups of a session, keyed by group id (the hyp_id of its first test); a group is never deleted, only its
    level and status change."""

    def __init__(self, n: int, rows=()):
        self.n, self.groups, self._keys = n, {}, {}
        for r in rows:
            g = dict(r)
            for k in COUNTER.values():
                g[k] = int(g[k])
            g["evidence"], g["exceptions"] = g["evidence"].split(), g["exceptions"].split()
            self.groups[g["group_id"]] = g
            self._keys[self.key(g["region"], g["prediction"])] = g["group_id"]

    def key(self, region: str, prediction: str) -> tuple:
        """Two regions are the same if they select the same set of canonical sensor_layouts at Σ = n."""
        return hypothesis.region_mask(hypothesis.parse(region), self.n).tobytes(), prediction

    def rows(self) -> list[dict]:
        return [dict(g, evidence=" ".join(g["evidence"]), exceptions=" ".join(g["exceptions"]))
                for g in self.groups.values()]

    def level(self, level: str) -> list[dict]:
        return [g for g in self.groups.values() if g["level"] == level]

    def recent_short(self) -> list[dict]:
        """The MAX_SHORT groups that are not rules, with the latest tests."""
        last = lambda g: _batch_no(g["evidence"][-1] if g["evidence"] else g["group_id"])
        return sorted(self.level("short"), key=last)[-MAX_SHORT:]

    def flagged(self) -> list[str]:
        return [g["group_id"] for g in self.level("long") if g["flag"]]

    def find(self, region: str, prediction: str) -> str | None:
        return self._keys.get(self.key(region, prediction))

    def obeys(self, group_id: str, counts):
        """(M,) mask: a rule with prediction "lower" prefers its region; with "higher", the rest."""
        g = self.groups[group_id]
        inside = hypothesis.holds_mirror(hypothesis.parse(g["region"]), counts)
        return inside if g["prediction"] == "lower" else ~inside

    # ------------------------------------------------------------------ flag decisions (M9)

    def check_decisions(self, decisions) -> list[str]:
        errs, flagged, given = [], set(self.flagged()), set()
        for d in decisions:
            rid = d["rule_id"].strip()
            if d["decision"] not in ("keep", "demote"):
                errs.append(f"flag decision for {rid}: decision {d['decision']!r} is not keep or demote")
            if rid not in flagged:
                errs.append(f"flag decision for {rid!r}: it is not a flagged rule")
            elif rid in given:
                errs.append(f"flag decision for {rid}: given twice")
            elif not d["reason"].strip():
                errs.append(f"flag decision for {rid}: give a reason")
            given.add(rid)
        return errs + [f"rule {rid} is flagged: give a keep or demote decision with a reason"
                       for rid in sorted(flagged - given)]

    def apply_decisions(self, decisions, hyp_id: str):
        for d in decisions:
            g = self.groups[d["rule_id"].strip()]
            g["flag"] = ""
            g["last_reason"] = f"{hyp_id}: {d['decision']}: {' '.join(d['reason'].split())}"
            if d["decision"] == "demote":
                g["level"] = "short"

    # ------------------------------------------------------------------ harness updates

    def assign(self, region: str, prediction: str, hyp_id: str) -> str:
        """Group id of the hypothesis; a new group gets the id hyp_id (design M3)."""
        gid = self.find(region, prediction)
        if gid is None:
            gid = hyp_id
            self.groups[gid] = dict(group_id=gid, level="short", region=region, prediction=prediction,
                                    status="open", n_support=0, n_refute=0, n_inconclusive=0, evidence=[],
                                    exceptions=[], flag="", last_reason=f"first test {hyp_id}")
            self._keys[self.key(region, prediction)] = gid
        return gid

    def record_test(self, group_id: str, hyp_id: str, verdict: str):
        g = self.groups[group_id]
        g[COUNTER[verdict]] += 1
        g["status"] = verdict
        g["evidence"].append(hyp_id)

    def record_exception(self, rule_id: str, hyp_id: str):
        self.groups[rule_id]["exceptions"].append(hyp_id)

    def promote(self, group_id: str, n_promote: int, hyp_id: str) -> list[str]:
        """After a supported test: n_support >= n_promote and n_refute = 0 -> rule, while the long-term list has fewer
        than MAX_LONG rules. Only the tested group is checked, so a demoted rule returns only after a new support."""
        g = self.groups[group_id]
        if g["level"] != "short" or g["status"] != "supported" or g["n_support"] < n_promote or g["n_refute"]:
            return []
        if len(self.level("long")) >= MAX_LONG:
            g["last_reason"] = f"{hyp_id}: promotion blocked, the long-term list has {MAX_LONG} rules"
            return [f"Group {group_id}: promotion blocked, the long-term list is full."]
        g.update(level="long", last_reason=f"promoted at {hyp_id}")
        return [f"Group {group_id} promoted to the long-term list ({g['n_support']} supported tests, none refuted)."]

    def update_flags(self, hyp: dict, group_id: str | None, batch_rows, valid_rows, tau: float) -> list[str]:
        """Flag a rule if its test in this batch gave refuted, or if a valid trial of this batch breaks the rule and
        beats the best valid trial that obeys it by more than τ (design M8)."""
        lines, new_ids = [], {r["id"] for r in batch_rows}
        for g in self.level("long"):
            reasons = []
            if group_id == g["group_id"] and hyp["verdict"] == "refuted":
                reasons.append(f"its test {hyp['hyp_id']} gave refuted")
            ok = {r["id"]: bool(self.obeys(g["group_id"], layout.parse(r["config"]))[0]) for r in valid_rows}
            obey = [r for r in valid_rows if ok[r["id"]]]
            brk = [r for r in valid_rows if r["id"] in new_ids and not ok[r["id"]]]
            if obey and brk:
                b, o = min(brk, key=lambda r: r["score"]), min(obey, key=lambda r: r["score"])
                if o["score"] - b["score"] > tau:
                    reasons.append(f"{b['id']} ({b['config']}, score {state.show('score', b['score'])}) breaks the "
                                   f"rule and beats the best trial that obeys it, {o['id']} (score "
                                   f"{state.show('score', o['score'])}), by {o['score'] - b['score']:.4f} > "
                                   f"τ = {tau:.4f}")
            if reasons:
                g["flag"] = "; ".join(reasons)
                lines.append(f"Rule {g['group_id']} flagged: {g['flag']}.")
        return lines
