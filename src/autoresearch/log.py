"""The log (results.csv, one row per trial), calls.csv (one row per planner call), the session cache, and the files
of the learnings run: starts.csv, hypotheses.csv, learnings.csv, confirm.csv, confirm_starts.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from . import layout

RESULT_COLUMNS = ("id", "config", "score", "s1", "s2", "rho", "n_iter", "converged", "wall_time",
                  "evaluator_hash", "status", "description")
CALL_COLUMNS = ("call_id", "retry", "wall_time", "input_tokens", "cache_read_tokens", "cache_write_tokens",
                "output_tokens", "cost", "model", "stop_reason")
START_COLUMNS = ("id", "start", "seed", "phi_train", "phi_val", "s1", "s2", "rho", "n_iter", "converged",
                 "min_spacing_mm", "valid", "selected", "wall_time")
HYP_COLUMNS = ("hyp_id", "parent_id", "step", "region", "context", "prediction", "ref_id", "n_set", "B_h",
               "reused_ids", "best_id", "n_cmp", "n_ref", "delta", "tau_h", "verdict", "branch", "exceptions",
               "rationale")
CONFIRM_COLUMNS = ("id", "config", "score_session", "score_confirm", "phi_train_confirm", "s1", "s2", "rho",
                   "n_valid", "wall_time")
FLOATS = ("score", "s1", "s2", "rho", "wall_time")
START_FLOATS = ("phi_train", "phi_val", "s1", "s2", "rho", "min_spacing_mm", "wall_time")
HYP_INTS = ("n_set", "B_h", "n_cmp", "n_ref")


def _append(path: Path, columns, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})
        fh.flush()


def _write(path: Path, columns, rows):
    """Full rewrite through a temporary file, so an interrupted write leaves the previous version."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({c: "" if row.get(c) is None else row.get(c) for c in columns})
    tmp.replace(path)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _float(v):
    return float(v) if v not in ("", None) else None


def parse_result(row: dict) -> dict:
    """CSV strings -> typed values; empty metric fields of crashed trials stay None."""
    out = dict(row)
    for c in FLOATS:
        out[c] = _float(row.get(c))
    out["n_iter"] = int(row["n_iter"]) if row.get("n_iter") not in ("", None) else None
    out["converged"] = row.get("converged") == "True"
    return out


def parse_start(row: dict) -> dict:
    """starts.csv strings -> typed values; empty metric fields of crashed starts stay None."""
    out = dict(row)
    for c in START_FLOATS:
        out[c] = _float(row.get(c))
    for c in ("start", "seed"):
        out[c] = int(row[c])
    out["n_iter"] = int(row["n_iter"]) if row.get("n_iter") not in ("", None) else None
    for c in ("converged", "valid", "selected"):
        out[c] = row.get(c) == "True"
    return out


def parse_hypothesis(row: dict) -> dict:
    out = dict(row)
    for c in ("delta", "tau_h"):
        out[c] = _float(row.get(c))
    for c in HYP_INTS:
        out[c] = int(row[c]) if row.get(c) not in ("", None) else 0
    return out


class Log:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.results_path = self.folder / "results.csv"
        self.calls_path = self.folder / "calls.csv"
        self.starts_path = self.folder / "starts.csv"
        self.hyp_path = self.folder / "hypotheses.csv"
        self.learnings_path = self.folder / "learnings.csv"
        self.confirm_path = self.folder / "confirm.csv"
        self.confirm_starts_path = self.folder / "confirm_starts.csv"

    def append_result(self, row: dict):
        _append(self.results_path, RESULT_COLUMNS, row)

    def append_call(self, row: dict):
        _append(self.calls_path, CALL_COLUMNS, row)

    def append_starts(self, rows, confirm=False):
        for row in rows:
            _append(self.confirm_starts_path if confirm else self.starts_path, START_COLUMNS, row)

    def append_confirm(self, row: dict):
        _append(self.confirm_path, CONFIRM_COLUMNS, row)

    def write_hypotheses(self, rows):
        _write(self.hyp_path, HYP_COLUMNS, rows)

    def write_learnings(self, rows, columns):
        _write(self.learnings_path, columns, rows)

    def results(self) -> list[dict]:
        return [parse_result(r) for r in _read(self.results_path)]

    def calls(self) -> list[dict]:
        return _read(self.calls_path)

    def starts(self) -> list[dict]:
        return [parse_start(r) for r in _read(self.starts_path)]

    def hypotheses(self) -> list[dict]:
        return [parse_hypothesis(r) for r in _read(self.hyp_path)]

    def learnings(self) -> list[dict]:
        return _read(self.learnings_path)

    def confirm(self) -> list[dict]:
        return _read(self.confirm_path)

    def confirm_starts(self) -> list[dict]:
        return [parse_start(r) for r in _read(self.confirm_starts_path)]


def cache_key(config: str, mirror_reduce: bool) -> tuple[int, ...]:
    counts = layout.parse(config)
    return layout.canonical(counts) if mirror_reduce else counts


def build_cache(rows, mirror_reduce: bool) -> dict:
    """cache key -> the first logged row of that sensor_layout."""
    cache = {}
    for r in rows:
        cache.setdefault(cache_key(r["config"], mirror_reduce), r)
    return cache
