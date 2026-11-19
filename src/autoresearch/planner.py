"""Planners: the LLM planner through the local proxy, a seeded random planner, the exhaustive enumerator, and a fake
planner with fixed replies for the learnings run."""

from __future__ import annotations

import json
import time
from typing import NamedTuple

import numpy as np

from . import hypothesis, layout
from . import problem as P

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"config": {"type": "string"}, "description": {"type": "string"}},
                "required": ["config", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
# property order is the generation order of the reply (design X2): flag decisions, hypothesis
LEARN_SCHEMA = _obj(
    flag_decisions={"type": "array", "items": _obj(
        rule_id=_STR, decision={"type": "string", "enum": ["keep", "demote"]}, reason=_STR)},
    hypothesis=_obj(
        parent_id=_STR, step={"type": "string", "enum": list(hypothesis.STEPS)}, region=_STR, context=_STR,
        prediction={"type": "string", "enum": list(hypothesis.PREDICTIONS)}, ref_id=_STR, rationale=_STR,
        exceptions={"type": "array", "items": _obj(rule_id=_STR, reason=_STR)}),
)


class Reply(NamedTuple):
    proposals: list  # (config, description) pairs, unchecked
    call: dict | None  # calls.csv row, None for planners without an API call
    raw: dict | None  # request settings and full response, saved for audit


class LearnReply(NamedTuple):
    data: dict | None  # parsed JSON reply, unchecked; None if there is none
    error: str | None  # why data is None
    content: list | None  # assistant content blocks to send back in a retry; None if a retry is not possible
    call: dict  # calls.csv row
    raw: dict | None  # request settings and full response, saved for audit


class AnthropicPlanner:
    """Claude through the proxy; the proxy holds the API key, so the client key is a placeholder."""

    def __init__(self, program: str, *, model: str, effort: str, base_url: str, prices: dict,
                 max_tokens: int = 16000, cache_ttl: str = "5m", max_retries: int = 4):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(base_url=base_url, api_key="via-proxy", max_retries=max_retries)
        self.program, self.model, self.effort = program, model, effort
        self.prices, self.max_tokens, self.cache_ttl = prices, max_tokens, cache_ttl
        self.n_calls = 0

    def cost(self, usage) -> float:
        """USD from the usage fields and the per-MTok prices."""
        p = self.prices
        return 1e-6 * (usage["input_tokens"] * p["input"] + usage["cache_write_tokens"] * p["cache_write"]
                       + usage["cache_read_tokens"] * p["cache_read"] + usage["output_tokens"] * p["output"])

    def _send(self, messages: list, schema: dict, call_id: str, retry: int):
        """One API call -> (response or None after an API error, calls.csv row, audit dict)."""
        cache = {"type": "ephemeral"} if self.cache_ttl == "5m" else {"type": "ephemeral", "ttl": self.cache_ttl}
        request = dict(model=self.model, max_tokens=self.max_tokens,
                       system=[{"type": "text", "text": self.program, "cache_control": cache}],
                       messages=messages, thinking={"type": "adaptive", "display": "summarized"},
                       output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                       betas=[FALLBACK_BETA], fallbacks="default")
        self.n_calls += 1
        t0 = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(**request)
        except self.anthropic.APIError as e:
            wall = time.perf_counter() - t0
            call = dict(call_id=call_id, retry=retry, wall_time=wall, input_tokens=0, cache_read_tokens=0,
                        cache_write_tokens=0, output_tokens=0, cost=0.0, model=self.model,
                        stop_reason=f"error:{type(e).__name__}")
            return None, call, dict(request=_audit(request), error=repr(e))
        wall = time.perf_counter() - t0
        u = resp.usage
        usage = dict(input_tokens=u.input_tokens or 0, cache_read_tokens=u.cache_read_input_tokens or 0,
                     cache_write_tokens=u.cache_creation_input_tokens or 0, output_tokens=u.output_tokens or 0)
        call = dict(call_id=call_id, retry=retry, wall_time=wall, **usage, cost=self.cost(usage), model=resp.model,
                    stop_reason=resp.stop_reason)
        return resp, call, dict(request=_audit(request), response=resp.to_dict())

    def propose(self, state_text: str, call_id: str) -> Reply:
        resp, call, raw = self._send([{"role": "user", "content": state_text}], REPLY_SCHEMA, call_id, 0)
        proposals = []
        if resp is not None and resp.stop_reason not in ("refusal", "max_tokens"):
            text = next((b.text for b in resp.content if b.type == "text"), "")
            try:
                proposals = [(p["config"], p["description"]) for p in json.loads(text)["proposals"]]
            except (ValueError, KeyError, TypeError) as e:
                raw["parse_error"] = repr(e)
        return Reply(proposals, call, raw)

    def respond(self, messages: list, call_id: str, retry: int) -> LearnReply:
        """One learn-mode call on the conversation messages (state, then earlier replies and their error lists)."""
        resp, call, raw = self._send(messages, LEARN_SCHEMA, call_id, retry)
        if resp is None:
            return LearnReply(None, f"API error: {raw['error']}", None, call, raw)
        if resp.stop_reason in ("refusal", "max_tokens"):
            return LearnReply(None, f"stop reason {resp.stop_reason}", None, call, raw)
        blocks = raw["response"]["content"]
        cut = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
        # blocks before a fallback boundary are not sent back (refusal-fallback rule); null fields are not input fields
        content = [{k: v for k, v in b.items() if v is not None} for b in blocks[cut + 1:]]
        text = next((b["text"] for b in content if b.get("type") == "text"), "")
        try:
            data = json.loads(text)
        except ValueError as e:
            return LearnReply(None, f"the reply is not JSON: {e}", content, call, raw)
        return LearnReply(data, None, content, call, raw)


def _audit(request: dict) -> dict:
    """Request settings without the long texts, which are the program file and the state file."""
    return {k: v for k, v in request.items() if k not in ("system", "messages")}


class RandomPlanner:
    """Uniform random sensor_layouts; optionally repeats the first logged config in one call (smoke test)."""

    def __init__(self, n: int, b: int, seed: int, repeat_call: int | None = None):
        self.n, self.b, self.rng, self.repeat_call = n, b, np.random.default_rng(seed), repeat_call
        self.n_calls = 0
        self.first_config = None

    def propose(self, state_text: str, call_id: str) -> Reply:
        out = []
        for _ in range(self.b):
            cuts = np.sort(self.rng.choice(self.n + len(P.FACES) - 1, len(P.FACES) - 1, replace=False))
            bounds = np.concatenate([[-1], cuts, [self.n + len(P.FACES) - 1]])
            out.append((layout.fmt(np.diff(bounds) - 1), "random"))
        if self.n_calls == 0:
            self.first_config = out[0][0]
        if self.repeat_call is not None and self.n_calls == self.repeat_call:
            out[0] = (self.first_config, "forced repeat")
        self.n_calls += 1
        return Reply(out, None, None)


class FakeLearnPlanner:
    """Fixed learn-mode replies, no API: each reply is a dict, or a callable of the conversation messages that returns
    one; the call row has zero tokens and cost."""

    def __init__(self, replies):
        self.replies, self.n_calls = list(replies), 0

    def respond(self, messages: list, call_id: str, retry: int) -> LearnReply:
        call = dict(call_id=call_id, retry=retry, wall_time=0.0, input_tokens=0, cache_read_tokens=0,
                    cache_write_tokens=0, output_tokens=0, cost=0.0, model="fake", stop_reason="end_turn")
        if self.n_calls >= len(self.replies):
            self.n_calls += 1
            return LearnReply(None, "no fixed reply left", None, call, None)
        r = self.replies[self.n_calls]
        self.n_calls += 1
        data = r(messages) if callable(r) else r
        return LearnReply(data, None, [{"type": "text", "text": json.dumps(data)}], call, dict(reply=data))


class Enumerator:
    """All sensor_layouts in a seeded random order, in chunks of b; skips the ones already done."""

    def __init__(self, n: int, b: int, seed: int, mirror_reduce: bool, done=()):
        todo = layout.enumerate_layouts(n, mirror_reduce)
        order = np.random.default_rng(seed).permutation(len(todo))
        done = set(done)
        self.todo = [todo[i] for i in order if todo[i] not in done]
        self.b, self.pos = b, 0

    def propose(self, state_text: str, call_id: str) -> Reply:
        chunk = self.todo[self.pos:self.pos + self.b]
        self.pos += len(chunk)
        return Reply([(layout.fmt(c), "exhaustive") for c in chunk], None, None)
