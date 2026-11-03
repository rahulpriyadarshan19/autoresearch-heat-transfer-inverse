"""ID case2_ar_smoke_harness: the session loop end to end, with no API call.

Context: scripts/run_session.py runs src/autoresearch/harness.py with the LLM planner, the enumerator or a
random planner. This test runs, in a throwaway git repository inside its output folder:
    A  fake planner, N = 10, B = 4, 3 batches, maxiter 20, one forced repeat in batch 2
    B  exhaustive run, N = 10 (336 mirror-reduced sensor_layouts), maxiter 20, commit at the end
    C  exhaustive run, N = 3, stopped after 2 batches, then resumed to the end
    D  scripts/run_session.sh (GPU 0, CPUs 0,1,4,5) in fake mode, N = 10, 2 batches, maxiter 5
and checks the log columns, the id format, the statuses, the keep rule, the cache, the resume and the commits.
Then it runs scripts/plot_session.py on A and B (plots W1–W6).
Outputs: outputs/case2_ar_smoke_harness_<stamp>/{checks.txt, repo/sessions/, outputs/}.

Run:  venv_ar/bin/python scripts/smoke_test/smoke_ar_harness.py
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import time

import jax

from autoresearch import harness, layout, log, planner
from autoresearch import problem as P
from heat_inverse import config


class Checks:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, name, passed, detail=""):
        self.ok &= bool(passed)
        self.lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def info(self, text):
        self.lines.append(f"INFO  {text}")


def keep_rule_holds(rows, notes) -> bool:
    best = None
    for r in rows:
        ok = r["score"] is not None and notes.get(r["id"]) != "infeasible"
        expect = "keep" if ok and (best is None or r["score"] < best) else ("crash" if r["score"] is None
                                                                             else "discard")
        if r["status"] != expect:
            return False
        if expect == "keep":
            best = r["score"]
    return True


def check_log(c, name, session, n_expected):
    rows = session.log.results()
    header = session.log.results_path.read_text().splitlines()[0].split(",")
    c.add(f"{name}: results.csv columns", tuple(header) == log.RESULT_COLUMNS, ",".join(header))
    c.add(f"{name}: id format", all(re.fullmatch(r"b\d{3}-\d{2}", r["id"]) for r in rows))
    c.add(f"{name}: statuses", {r["status"] for r in rows} <= {"keep", "discard", "crash"},
          str(sorted({r["status"] for r in rows})))
    c.add(f"{name}: keep rule", keep_rule_holds(rows, session.notes))
    keys = [log.cache_key(r["config"], session.mirror_reduce) for r in rows]
    c.add(f"{name}: no sensor_layout run twice", len(keys) == len(set(keys)), f"{len(rows)} rows")
    c.add(f"{name}: number of trials", len(rows) == n_expected, f"{len(rows)} vs {n_expected}")
    return rows


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_harness"))
    repo, outputs = out / "repo", out / "outputs"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "smoke"), ("config", "user.email", "smoke@localhost")):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    c, t_all = Checks(), time.perf_counter()
    banks = P.load_banks()
    common = dict(root=repo, out_root=outputs, allow_dirty=True, banks=banks, budget_s=math.inf, m=5,
                  maxiter=20, kill_time=None)

    t0 = time.perf_counter()
    a = harness.Session(session_id="n10_fake_smoke", mode="fake", n=10, b=4, mirror_reduce=True,
                        planner=planner.RandomPlanner(10, 4, seed=0, repeat_call=1), commit_mode="batch",
                        max_batches=3, **common)
    meta_a = a.run()
    c.info(f"A: {time.perf_counter() - t0:.1f} s")
    replay = planner.RandomPlanner(10, 4, seed=0, repeat_call=1)
    proposed = [layout.parse(cfg) for i in range(3) for cfg, _ in replay.propose("", f"c{i}").proposals]
    unique = {layout.canonical(p) for p in proposed}
    c.info(f"A: {len(proposed)} proposals, {len(unique)} distinct up to mirror images")
    rows_a = check_log(c, "A", a, len(unique))
    c.add("A: forced repeat not run, batch 2 has 3 trials", sum(r["id"].startswith("b002") for r in rows_a) == 3)
    run_cfgs = {r["config"] for r in rows_a}
    mirrors = [layout.fmt(p) for p in proposed if layout.fmt(p) not in run_cfgs]
    c.info(f"A: proposals not run (repeat or mirror image of a run one): {', '.join(mirrors)}")
    c.add("A: stop reason", meta_a["stop"] == "max batches", meta_a["stop"])

    t0 = time.perf_counter()
    b = harness.Session(session_id="n10_exh_smoke", mode="exh", n=10, b=16, mirror_reduce=True,
                        planner=planner.Enumerator(10, 16, seed=0, mirror_reduce=True), commit_mode="end",
                        **common)
    meta_b = b.run()
    c.info(f"B: {time.perf_counter() - t0:.1f} s for {meta_b['n_trials']} trials")
    check_log(c, "B", b, len(layout.enumerate_layouts(10)))
    c.add("B: stop reason", meta_b["stop"] == "all sensor_layouts tried", meta_b["stop"])

    c1 = harness.Session(session_id="n3_exh_smoke", mode="exh", n=3, b=4, mirror_reduce=True,
                         planner=planner.Enumerator(3, 4, seed=1, mirror_reduce=True), commit_mode="end",
                         max_batches=2, **common)
    c1.run()
    done = log.build_cache(c1.log.results(), True)
    c2 = harness.Session(session_id="n3_exh_smoke", mode="exh", n=3, b=4, mirror_reduce=True,
                         planner=planner.Enumerator(3, 4, seed=1, mirror_reduce=True, done=done), commit_mode="end",
                         resume=True, **common)
    meta_c = c2.run()
    rows_c = check_log(c, "C (resumed)", c2, len(layout.enumerate_layouts(3)))
    c.add("C: ids continue after the resume", rows_c[8]["id"].startswith("b003"), rows_c[8]["id"])
    c.add("C: stop reason", meta_c["stop"] == "all sensor_layouts tried", meta_c["stop"])

    n_commits = int(subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    # A: 3 batch commits + 1 end commit; B: 1 end commit; C: 1 end commit before and 1 after the resume
    c.add("commits", n_commits == 7, f"{n_commits} commits")
    clean = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip() == ""
    c.add("working tree clean after the sessions", clean)
    c.info(f"evaluator hash in the log: {rows_a[0]['evaluator_hash']} (the code is not yet in git)")

    t0 = time.perf_counter()
    r = subprocess.run([str(config.PROJECT_ROOT / "scripts" / "run_session.sh"), "0", "0,1,4,5", "--mode", "fake",
                        "--n", "10", "--b", "4", "--max-batches", "2", "--maxiter", "5", "--session-id", "n10_fake_cli",
                        "--root", str(repo), "--out-root", str(outputs), "--allow-dirty"],
                       capture_output=True, text=True)
    rows_d = log.Log(repo / "sessions" / "n10_fake_cli").results()
    c.add("D: run_session.sh in fake mode", r.returncode == 0 and len(rows_d) > 0,
          f"{len(rows_d)} trials in {time.perf_counter() - t0:.1f} s" + ("" if r.returncode == 0 else r.stderr[-800:]))
    n_commits_d = int(subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"], capture_output=True,
                                     text=True, check=True).stdout)
    c.add("D: commits (2 batches + end)", n_commits_d == n_commits + 3, f"{n_commits_d - n_commits} new commits")

    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "scripts" / "plot_session.py"), "--llm",
                        "n10_fake_smoke", "--exh", "n10_exh_smoke", "--root", str(repo), "--out-root", str(outputs)],
                       capture_output=True, text=True)
    plots = sorted(p.name for p in (outputs / "n10_fake_smoke" / "plots").glob("*.png"))
    c.add("plot_session.py on A and B", r.returncode == 0 and len(plots) == 7,
          f"{len(plots)} plots in {time.perf_counter() - t0:.1f} s" + ("" if r.returncode == 0 else r.stderr[-500:]))

    c.info(f"total wall time {time.perf_counter() - t_all:.1f} s")
    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print("\n".join(c.lines), f"\n-> {out}", sep="")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
