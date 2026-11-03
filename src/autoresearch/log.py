"""The log (results.csv, one row per trial), calls.csv (one row per planner call) and the session cache."""

from __future__ import annotations

import csv
from pathlib import Path

from . import layout

RESULT_COLUMNS = ("id", "config", "score", "s1", "s2", "rho", "n_iter", "converged", "wall_time",
                  "evaluator_hash", "status", "description")
CALL_COLUMNS = ("call_id", "wall_time", "input_tokens", "cache_read_tokens", "cache_write_tokens",
                "output_tokens", "cost", "model", "stop_reason")
FLOATS = ("score", "s1", "s2", "rho", "wall_time")


def _append(path: Path, columns, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})
        fh.flush()


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def parse_result(row: dict) -> dict:
    """CSV strings -> typed values; empty metric fields of crashed trials stay None."""
    out = dict(row)
    for c in FLOATS:
        out[c] = float(row[c]) if row.get(c) not in ("", None) else None
    out["n_iter"] = int(row["n_iter"]) if row.get("n_iter") not in ("", None) else None
    out["converged"] = row.get("converged") == "True"
    return out


class Log:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.results_path = self.folder / "results.csv"
        self.calls_path = self.folder / "calls.csv"

    def append_result(self, row: dict):
        _append(self.results_path, RESULT_COLUMNS, row)

    def append_call(self, row: dict):
        _append(self.calls_path, CALL_COLUMNS, row)

    def results(self) -> list[dict]:
        return [parse_result(r) for r in _read(self.results_path)]

    def calls(self) -> list[dict]:
        return _read(self.calls_path)


def cache_key(config: str, mirror_reduce: bool) -> tuple[int, ...]:
    counts = layout.parse(config)
    return layout.canonical(counts) if mirror_reduce else counts


def build_cache(rows, mirror_reduce: bool) -> dict:
    """cache key -> the first logged row of that sensor_layout."""
    cache = {}
    for r in rows:
        cache.setdefault(cache_key(r["config"], mirror_reduce), r)
    return cache
