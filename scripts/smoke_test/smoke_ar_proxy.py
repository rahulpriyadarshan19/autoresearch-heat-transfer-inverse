"""ID case2_ar_smoke_proxy: the proxy and the planner call, after `sudo proxy/install_proxy.sh`.

Context: the API key is only in /etc/llmproxy/key (user llmkey, mode 400); the harness calls
http://127.0.0.1:8765, and the proxy adds the key. This test checks, as the user that runs the harness:
    no API key in the environment; the key file and /etc/llmproxy are not readable
    the proxy refuses a GET, an unknown path and a model outside its allowlist (no API cost)
    two real planner calls with program.md and an empty N = 30 state (B = 5): the reply parses, the
    proposals pass the harness checks, usage and cost are recorded, and the second call reads the cached
    system prompt (cache_read_tokens > 0). Cost: a few cents (estimate).
Outputs: outputs/case2_ar_smoke_proxy_<stamp>/{checks.txt, calls.csv, calls/*.json}.

Run:  venv_ar/bin/python scripts/smoke_test/smoke_ar_proxy.py [--effort high]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from autoresearch import harness, layout, log, planner, state
from heat_inverse import config

PROXY = "http://127.0.0.1:8765"
PRICES = dict(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0)  # claude-opus-5-5, USD per MTok


class Checks:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, name, passed, detail=""):
        self.ok &= bool(passed)
        self.lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def info(self, text):
        self.lines.append(f"INFO  {text}")


def status_of(req) -> int:
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()
    out = config.run_dir(harness.run_id("case2_ar_smoke_proxy"))
    c = Checks()

    leaked = [k for k, v in os.environ.items() if v.startswith("sk-ant") or k in ("ANTHROPIC_API_KEY",)]
    c.add("no API key in the environment", not leaked, ", ".join(leaked) or "none")
    c.add("key file not readable", not os.access("/etc/llmproxy/key", os.R_OK))
    c.add("/etc/llmproxy not listable", not os.access("/etc/llmproxy", os.R_OK | os.X_OK))

    c.add("GET refused", status_of(urllib.request.Request(PROXY + "/v1/messages")) == 405)
    c.add("unknown path refused", status_of(urllib.request.Request(PROXY + "/v1/models", data=b"{}")) == 404)
    body = json.dumps({"model": "not-allowed", "max_tokens": 1, "messages": []}).encode()
    c.add("model outside the allowlist refused",
          status_of(urllib.request.Request(PROXY + "/v1/messages", data=body,
                                           headers={"content-type": "application/json"})) == 403)

    program = (config.PROJECT_ROOT / "program.md").read_text()
    p = planner.AnthropicPlanner(program, model=args.model, effort=args.effort, base_url=PROXY, prices=PRICES)
    st = state.build([], {}, n=30, b=5, m=50, remaining_s=36000.0, untried=len(layout.enumerate_layouts(30)))
    lg = log.Log(out)
    (out / "calls").mkdir()
    for i in (1, 2):
        reply = p.propose(st, f"c{i:03d}")
        lg.append_call(reply.call)
        reply.raw["state"] = st
        (out / "calls" / f"c{i:03d}.json").write_text(json.dumps(reply.raw, indent=1, default=str))
        call = reply.call
        c.add(f"call {i}: reply", call["stop_reason"] == "end_turn" and len(reply.proposals) > 0,
              f"stop_reason {call['stop_reason']}, {len(reply.proposals)} proposals, model {call['model']}")
        valid = []
        for cfg, desc in reply.proposals:
            try:
                layout.parse(cfg, 30)
                valid.append(len(desc) <= 100)
            except ValueError:
                valid.append(False)
        c.add(f"call {i}: proposals pass the checks", reply.proposals and all(valid),
              "; ".join(f"{cfg} | {desc}" for cfg, desc in reply.proposals))
        c.info(f"call {i}: {call['wall_time']:.1f} s, input {call['input_tokens']}, cache write "
               f"{call['cache_write_tokens']}, cache read {call['cache_read_tokens']}, output "
               f"{call['output_tokens']} tokens, {call['cost']:.4f} USD")
    c.add("prompt caching of program.md", reply.call["cache_read_tokens"] > 0,
          f"cache read tokens in call 2: {reply.call['cache_read_tokens']}")

    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print("\n".join(c.lines), f"\n-> {out}", sep="")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
