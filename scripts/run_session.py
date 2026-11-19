"""ID case2_ar_session: one LLM session, learnings session, exhaustive run or fake-planner run for a fixed N.

Smoke tests: smoke_test/smoke_ar_harness.py runs this code path with the fake planner and the enumerator;
smoke_test/smoke_ar_proxy.py makes one planner call through the proxy; smoke_test/smoke_ar_pilot.py sets
maxiter, kill time, B and M. For --mode learn: smoke_test/smoke_ar_multistart.py (case2_ar_smoke_multistart) chose the
SciPy optimiser; smoke_test/smoke_ar_noise.py (case2_ar_smoke_noise) writes the calibration.json that --calibration
reads (R, σ_R, τ, t̄_start); smoke_test/smoke_ar_learn.py (case2_ar_smoke_learn) checks this flow with a fake planner.

Run:  scripts/run_session.sh <gpu> <cpus> --mode llm --n 30 --b 8 --kill-time 120
      scripts/run_session.sh <gpu> <cpus> --mode learn --n 25 --calibration outputs/case2_ar_smoke_noise_<stamp>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from autoresearch import harness, log, noise, planner
from autoresearch import problem as P
from heat_inverse import config


def _section(title, body):
    print(f"\n===== {title} =====\n{body}", flush=True)


class EchoPlanner(planner.AnthropicPlanner):
    """AnthropicPlanner that prints the system prompt once, then the user message and the response of each call."""

    def _echo(self, label, reply):
        if reply.raw is None or "error" in reply.raw:
            _section(f"{label} error", (reply.raw or {}).get("error", ""))
        else:
            for blk in reply.raw["response"]["content"]:
                _section(f"{label} {blk['type']}", blk.get("thinking", blk.get("text", "")))
        c = reply.call
        print(f"\n{label}: stop {c['stop_reason']}, {c['wall_time']:.1f} s, {c['output_tokens']} output tokens, "
              f"{c['cache_read_tokens']} cache-read tokens, {c['cost']:.4f} USD", flush=True)

    def propose(self, state_text, call_id):
        if self.n_calls == 0:
            _section("system prompt (same in every call)", self.program)
        _section(f"{call_id} user message", state_text)
        reply = super().propose(state_text, call_id)
        self._echo(call_id, reply)
        return reply

    def respond(self, messages, call_id, retry):
        if self.n_calls == 0:
            _section("system prompt (same in every call)", self.program)
        _section(f"{call_id} retry {retry} user message", messages[-1]["content"])
        reply = super().respond(messages, call_id, retry)
        self._echo(f"{call_id} retry {retry}", reply)
        return reply


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
    return EchoPlanner(Path(args.program).read_text(), model=args.model, effort=args.effort,
                       base_url=args.proxy_url, prices=prices, cache_ttl=args.cache_ttl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("llm", "learn", "exh", "fake"), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--b", type=int, default=None, help="batch size; llm, exh and fake only")
    ap.add_argument("--m", type=int, default=50, help="best rows in the state")
    ap.add_argument("--maxiter", type=int, default=None, help="default 1000; learn: from the calibration")
    ap.add_argument("--kill-time", type=float, default=None, help="s per trial (per start with --r or learn); "
                                                                    "learn: from the calibration")
    ap.add_argument("--budget-h", type=float, default=None, help="default 10 h, none for --mode exh")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="fake planner and enumerator order")
    ap.add_argument("--no-mirror-reduce", dest="mirror_reduce", action="store_false")
    ap.add_argument("--session-id", default=None, help="default n<N>_<mode>_<stamp>_<id>")
    ap.add_argument("--resume", action="store_true", help="exh only: continue an interrupted run")
    ap.add_argument("--commit", choices=("batch", "end", "none"), default=None,
                    help="default batch for llm, learn and fake, end for exh")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only")
    ap.add_argument("--root", type=Path, default=config.PROJECT_ROOT, help="repository that holds sessions/")
    ap.add_argument("--out-root", type=Path, default=config.OUTPUTS)
    ap.add_argument("--program", default=None, help="default program.md; learn: program_learn.md")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--proxy-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cache-ttl", default="5m", choices=("5m", "1h"))
    ap.add_argument("--price-input", type=float, default=4.0, help="USD per MTok")
    ap.add_argument("--price-output", type=float, default=20.0)
    ap.add_argument("--price-cache-read", type=float, default=0.20)
    ap.add_argument("--price-cache-write", type=float, default=5.0)
    ap.add_argument("--r", type=int, default=None, help="starts per sensor_layout, seeds 0…R−1; learn: default "
                                                        "the calibrated R; exh: default one start, v1 rule")
    ap.add_argument("--calibration", type=Path, default=None, help="learn: case2_ar_smoke_noise output folder")
    ap.add_argument("--b-max", type=int, default=None, help="learn: default min(32, ⌊0.05 T / (R t̄_start)⌋)")
    ap.add_argument("--k-stop", type=int, default=5, help="learn: batches with no gain > ε_eff before the stop")
    ap.add_argument("--eps", type=float, default=noise.EPS_GAIN, help="learn: smallest gain of interest ε")
    ap.add_argument("--n-promote", type=int, default=2, help="learn: supported tests for promotion")
    ap.add_argument("--screen", default=harness.SCREEN, help="learn: constraint of batch 1")
    ap.add_argument("--n-confirm", type=int, default=5, help="learn: top sensor_layouts run again after the stop")
    ap.add_argument("--max-retries", type=int, default=2, help="learn: corrected replies per call")
    args = ap.parse_args()
    if args.resume and args.mode != "exh":
        ap.error("--resume is for --mode exh only")
    if args.resume and not args.session_id:
        ap.error("--resume needs --session-id")
    if args.mode != "learn" and args.b is None:
        ap.error(f"--mode {args.mode} needs --b")
    if args.r is not None and args.mode not in ("learn", "exh"):
        ap.error("--r is for --mode learn and exh")
    args.program = args.program or str(config.PROJECT_ROOT / ("program_learn.md" if args.mode == "learn"
                                                              else "program.md"))
    args.session_id = args.session_id or harness.run_id(f"n{args.n}_{args.mode}")
    budget_h = args.budget_h if args.budget_h is not None else (math.inf if args.mode == "exh" else 10.0)
    commit_mode = args.commit or ("end" if args.mode == "exh" else "batch")

    settings = dict(planner_mode=args.mode, seed=args.seed)
    if args.mode in ("llm", "learn"):
        settings.update(model=args.model, effort=args.effort, proxy_url=args.proxy_url, cache_ttl=args.cache_ttl,
                        program=args.program, program_sha256=P.file_sha256(args.program),
                        prices_usd_per_mtok=dict(input=args.price_input, output=args.price_output,
                                                 cache_read=args.price_cache_read,
                                                 cache_write=args.price_cache_write))
    common = dict(session_id=args.session_id, n=args.n, m=args.m, budget_s=3600 * budget_h,
                  planner=make_planner(args, args.root), root=args.root, out_root=args.out_root,
                  commit_mode=commit_mode, allow_dirty=args.allow_dirty, max_batches=args.max_batches,
                  settings=settings)
    if args.mode == "learn":
        if args.calibration is None:
            ap.error("--mode learn needs --calibration")
        cal = noise.load_calibration(args.calibration, R=args.r, eps=args.eps)
        if cal["N"] != args.n:
            ap.error(f"the calibration is for N = {cal['N']}, not {args.n}")
        maxiter = cal["maxiter"] if args.maxiter is None else args.maxiter
        kill_time = cal["kill_time"] if args.kill_time is None else args.kill_time
        if (maxiter, kill_time) != (cal["maxiter"], cal["kill_time"]):
            ap.error(f"maxiter and kill time must match the calibration ({cal['maxiter']}, {cal['kill_time']} s)")
        b_max = args.b_max or noise.b_max(3600 * budget_h, cal["R"], cal["t_start"])
        session = harness.LearnSession(
            noise_cal=cal, b_max=b_max, k_stop=args.k_stop, n_promote=args.n_promote, screen=args.screen,
            n_confirm=args.n_confirm, max_retries=args.max_retries, maxiter=maxiter, kill_time=kill_time, **common)
    else:
        session = harness.Session(
            mode=args.mode, b=args.b, maxiter=args.maxiter or 1000, kill_time=args.kill_time,
            mirror_reduce=args.mirror_reduce, resume=args.resume,
            seeds=range(args.r) if args.r is not None else None, **common)
    meta = session.run()
    keys = ("session_id", "stop", "elapsed_s", "n_trials", "n_calls", "cost_usd", "best_id", "best_config",
            "best_score", "confirm")
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
