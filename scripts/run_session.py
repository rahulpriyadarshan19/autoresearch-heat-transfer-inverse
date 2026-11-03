"""ID case2_ar_session: one LLM session, exhaustive run or fake-planner run for a fixed N.

Smoke tests: smoke_test/smoke_ar_harness.py runs this code path with the fake planner and the enumerator;
smoke_test/smoke_ar_proxy.py makes one planner call through the proxy; smoke_test/smoke_ar_pilot.py sets
maxiter, kill time, B and M.

Run:  scripts/run_session.sh <gpu> <cpus> --mode llm --n 30 --b 8 --kill-time 120
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from autoresearch import harness, log, planner
from autoresearch import problem as P
from heat_inverse import config


def make_planner(args, root):
    if args.mode == "fake":
        return planner.RandomPlanner(args.n, args.b, args.seed)
    if args.mode == "exh":
        done = []
        if args.resume:
            done = log.build_cache(log.Log(root / "sessions" / args.session_id).results(), args.mirror_reduce)
        return planner.Enumerator(args.n, args.b, args.seed, args.mirror_reduce, done=done)
    prices = dict(input=args.price_input, output=args.price_output, cache_read=args.price_cache_read,
                  cache_write=args.price_cache_write)
    return planner.AnthropicPlanner(Path(args.program).read_text(), model=args.model, effort=args.effort,
                                    base_url=args.proxy_url, prices=prices, cache_ttl=args.cache_ttl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("llm", "exh", "fake"), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--b", type=int, required=True, help="batch size")
    ap.add_argument("--m", type=int, default=50, help="best rows in the state")
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--kill-time", type=float, default=None, help="s per trial")
    ap.add_argument("--budget-h", type=float, default=None, help="default 10 h, none for --mode exh")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="fake planner and enumerator order")
    ap.add_argument("--no-mirror-reduce", dest="mirror_reduce", action="store_false")
    ap.add_argument("--session-id", default=None, help="default n<N>_<mode>_<stamp>_<id>")
    ap.add_argument("--resume", action="store_true", help="exh only: continue an interrupted run")
    ap.add_argument("--commit", choices=("batch", "end", "none"), default=None,
                    help="default batch for llm and fake, end for exh")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only")
    ap.add_argument("--root", type=Path, default=config.PROJECT_ROOT, help="repository that holds sessions/")
    ap.add_argument("--out-root", type=Path, default=config.OUTPUTS)
    ap.add_argument("--program", default=str(config.PROJECT_ROOT / "program.md"))
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--proxy-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cache-ttl", default="5m", choices=("5m", "1h"))
    ap.add_argument("--price-input", type=float, default=4.0, help="USD per MTok")
    ap.add_argument("--price-output", type=float, default=20.0)
    ap.add_argument("--price-cache-read", type=float, default=0.20)
    ap.add_argument("--price-cache-write", type=float, default=5.0)
    args = ap.parse_args()
    if args.resume and args.mode != "exh":
        ap.error("--resume is for --mode exh only")
    if args.resume and not args.session_id:
        ap.error("--resume needs --session-id")
    args.session_id = args.session_id or harness.run_id(f"n{args.n}_{args.mode}")
    budget_h = args.budget_h if args.budget_h is not None else (math.inf if args.mode == "exh" else 10.0)
    commit_mode = args.commit or ("end" if args.mode == "exh" else "batch")

    settings = dict(planner_mode=args.mode, seed=args.seed)
    if args.mode == "llm":
        settings.update(model=args.model, effort=args.effort, proxy_url=args.proxy_url, cache_ttl=args.cache_ttl,
                        program=args.program, program_sha256=P.file_sha256(args.program),
                        prices_usd_per_mtok=dict(input=args.price_input, output=args.price_output,
                                                 cache_read=args.price_cache_read,
                                                 cache_write=args.price_cache_write))
    session = harness.Session(
        session_id=args.session_id, mode=args.mode, n=args.n, b=args.b, m=args.m, maxiter=args.maxiter,
        kill_time=args.kill_time, budget_s=3600 * budget_h, mirror_reduce=args.mirror_reduce,
        planner=make_planner(args, args.root), root=args.root, out_root=args.out_root, commit_mode=commit_mode,
        resume=args.resume, allow_dirty=args.allow_dirty, max_batches=args.max_batches, settings=settings)
    meta = session.run()
    print(json.dumps({k: meta[k] for k in ("session_id", "stop", "elapsed_s", "n_trials", "n_calls", "cost_usd",
                                           "best_id", "best_config", "best_score")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
