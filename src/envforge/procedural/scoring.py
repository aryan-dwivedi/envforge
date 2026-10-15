"""Instance generation and response scoring through the sandbox runner.

Every place that turns (seed, difficulty) into a problem, or a model response
into a reward, goes through these two functions so gates, profiling, exports
and audits score identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from envforge.answer import extract_answer
from envforge.sandbox import Sandbox
from envforge.sandbox.harness import run_calls


@dataclass(frozen=True)
class Generated:
    seed: int
    difficulty: int
    problem: dict[str, Any] | None  # None when the call failed
    error: str | None = None
    message: str = ""


@dataclass(frozen=True)
class Score:
    value: float  # 0.0 whenever error is set or no answer was extracted
    error: str | None = None  # runner error kind from the check call, if any
    message: str = ""
    extracted: str | None = None


def generate(
    sb: Sandbox,
    module: str,
    pairs: list[tuple[int, int]],
    *,
    call_timeout_s: float,
    env: dict[str, str] | None = None,
) -> list[Generated]:
    """Call ``module.gen(seed, difficulty)`` for every pair, in order."""
    calls = [{"module": module, "fn": "gen", "args": [s, d]} for s, d in pairs]
    out = []
    for (s, d), r in zip(pairs, run_calls(sb, calls, call_timeout_s=call_timeout_s, env=env), strict=True):
        if r.get("ok"):
            out.append(Generated(s, d, r["value"]))
        else:
            out.append(Generated(s, d, None, r.get("error", "runner_crash"), r.get("message", "")))
    return out


def score(
    sb: Sandbox,
    module: str,
    items: list[tuple[dict[str, Any], str]],
    *,
    call_timeout_s: float,
    env: dict[str, str] | None = None,
) -> list[Score]:
    """Score each ``(problem, response_text)``.

    A response with no ``<answer>`` tag scores 0.0 without a check call.
    Otherwise the extracted answer goes to ``module.check``. A failed check
    call scores 0.0 and carries the runner error so callers that treat check
    errors as gate failures can see it.
    """
    extracted = [extract_answer(text) for _, text in items]
    idx = [i for i, a in enumerate(extracted) if a is not None]
    calls = [{"module": module, "fn": "check", "args": [items[i][0], extracted[i]]} for i in idx]
    results = run_calls(sb, calls, call_timeout_s=call_timeout_s, env=env) if calls else []
    out = [Score(0.0) for _ in items]
    for i, r in zip(idx, results, strict=True):
        if r.get("ok"):
            out[i] = Score(float(r["value"]), extracted=extracted[i])
        else:
            out[i] = Score(0.0, r.get("error", "runner_crash"), r.get("message", ""), extracted[i])
    return out
