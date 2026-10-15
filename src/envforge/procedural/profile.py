"""G9: difficulty profiling against a solver panel and the policy model."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import comb

from envforge.answer import task_messages
from envforge.llm import ChatModel, ChatResult
from envforge.procedural.scoring import generate, score
from envforge.sandbox import Sandbox
from envforge.schemas import GateResult, PassStats

PROFILE_SEEDS = range(1000, 1008)
MODULE = "generator.py"
K = 8


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def band(panel: list[PassStats], policy: PassStats) -> str:
    if sum(p.successes for p in panel) + policy.successes == 0:
        return "unsolved"
    if max((p.pass_at_1 for p in panel), default=0.0) > 0.9:
        return "too_easy"
    if policy.successes == 0:
        return "too_hard"
    return "target"


@dataclass(frozen=True)
class LevelProfile:
    difficulty: int
    panel: dict[str, PassStats]
    policy_key: str
    policy: PassStats
    band: str
    gate: GateResult


@dataclass(frozen=True)
class ProfileResult:
    levels: list[LevelProfile]
    cost_usd: float
    llm_errors: int


def _stats(c: int, n: int) -> PassStats:
    return PassStats(n=n, successes=c, k=K, pass_at_1=c / n, pass_at_k=pass_at_k(n, c, K))


def profile(
    base: Sandbox,
    *,
    levels: int,
    panel: list[tuple[str, ChatModel]],
    policy: ChatModel,
    call_timeout_s: float,
    max_tokens: int = 4096,
    concurrency: int = 8,
) -> ProfileResult:
    pairs = [(s, d) for d in range(1, levels + 1) for s in PROFILE_SEEDS]
    fork = base.fork()
    try:
        gens = generate(fork, MODULE, pairs, call_timeout_s=call_timeout_s)
    finally:
        fork.destroy()

    models: list[tuple[str, ChatModel]] = [(f"{fam}/{m.model}", m) for fam, m in panel]
    policy_key = f"policy/{policy.model}"
    models.append((policy_key, policy))

    # (model index, instance index) for every instance whose gen succeeded
    jobs = [(mi, gi) for mi in range(len(models)) for gi, g in enumerate(gens) if g.problem is not None]

    def ask(job: tuple[int, int]) -> ChatResult | None:
        mi, gi = job
        prompt = gens[gi].problem["prompt"]  # type: ignore[index]
        try:
            return models[mi][1].chat(task_messages(prompt), temperature=1.0, max_tokens=max_tokens)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(ask, jobs))

    cost = sum(r.cost_usd for r in replies if r is not None)
    errors = sum(1 for r in replies if r is None)
    answered = [(job, r) for job, r in zip(jobs, replies, strict=True) if r is not None]

    wins = [[0.0] * len(gens) for _ in models]
    if answered:
        fork = base.fork()
        try:
            items = [(gens[gi].problem, r.text) for (_, gi), r in answered]
            scores = score(fork, MODULE, items, call_timeout_s=call_timeout_s)  # type: ignore[arg-type]
        finally:
            fork.destroy()
        for ((mi, gi), _), s in zip(answered, scores, strict=True):
            wins[mi][gi] = s.value

    n = len(PROFILE_SEEDS)
    out = []
    for d in range(1, levels + 1):
        idx = [i for i, (_, gd) in enumerate(pairs) if gd == d]
        stats = {key: _stats(sum(1 for i in idx if wins[mi][i] >= 1.0), n) for mi, (key, _) in enumerate(models)}
        pol = stats.pop(policy_key)
        b = band(list(stats.values()), pol)
        gate = GateResult(
            passed=b == "target",
            metrics={
                "max_panel_pass1": max((p.pass_at_1 for p in stats.values()), default=0.0),
                "policy_pass_at_8": pol.pass_at_k,
                "llm_errors": float(errors),
            },
            detail=f"difficulty {d}: band {b}",
        )
        out.append(LevelProfile(d, stats, policy_key, pol, b, gate))
    return ProfileResult(out, cost, errors)
