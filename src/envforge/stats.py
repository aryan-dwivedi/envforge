"""Aggregate run statistics over a store."""

from __future__ import annotations

from collections import Counter
from typing import Any

from envforge.store import Store

GATE_ORDER = ("G1", "G2", "G3", "G4", "G5", "G10", "G9")


def compute_stats(store: Store) -> dict[str, Any]:
    """Summarise accepted environments, rejections, gates, bands, cost, splits and audits.

    An environment counts as accepted when it has at least one task in
    ``tasks/``; environments holding only regression tasks came from seeds
    whose ``RejectedRun`` already counts them.
    """
    tasks = store.load_tasks()
    regression = [t for t in store.load_tasks(include_regression=True) if t.task_id not in {x.task_id for x in tasks}]
    task_env = {t.task_id: t.env_id for t in tasks + regression}
    accepted_envs = sorted({t.env_id for t in tasks})
    accepted_set = set(accepted_envs)
    reports = store.load_reports()
    rejected = store.load_rejected()

    per_env: dict[str, Any] = {}
    for r in reports:
        env = task_env.get(r.task_id)
        if env in accepted_set and env not in per_env:
            per_env[env] = r

    gate_passes: Counter[str] = Counter()
    for r in per_env.values():
        gate_passes.update(g for g, res in r.gates.items() if res.passed)
    for rej in rejected:
        if rej.attempts:
            gate_passes.update(g for g, res in rej.attempts[-1].items() if res.passed)

    kill_rates = [r.mutant_kill_rate for r in per_env.values()]
    accepted_cost = sum(r.cost_usd for r in reports if task_env.get(r.task_id) in accepted_set)
    total_cost = accepted_cost + sum(rej.cost_usd for rej in rejected)
    band_of = {r.task_id: r.band for r in reports}
    audited = [r.human_audit for r in reports if r.human_audit is not None]

    return {
        "seeds_attempted": len(accepted_envs) + len(rejected),
        "accepted_envs": len(accepted_envs),
        "accepted_tasks": len(tasks),
        "regression_tasks": len(regression),
        "rejected_seeds": len(rejected),
        "gate_passes": {g: gate_passes[g] for g in GATE_ORDER},
        "bands": dict(sorted(Counter(band_of[t.task_id] for t in tasks + regression if t.task_id in band_of).items())),
        "mutant_kill_rate_mean": sum(kill_rates) / len(kill_rates) if kill_rates else None,
        "mutant_kill_rate_min": min(kill_rates) if kill_rates else None,
        "total_cost_usd": total_cost,
        "cost_per_accepted_task_usd": total_cost / len(tasks) if tasks else None,
        "splits": dict(sorted(Counter(t.split for t in tasks).items())),
        "audit": {
            "false_accepts": sum(a.false_accepts for a in audited),
            "n_audited": sum(a.n_audited for a in audited),
        },
    }


def format_stats(stats: dict[str, Any]) -> list[str]:
    """Render ``compute_stats`` output as printable lines."""

    def num(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    def counts(d: dict[str, int]) -> str:
        return ",".join(f"{k}:{v}" for k, v in d.items()) or "none"

    return [
        f"seeds attempted: {stats['seeds_attempted']} (accepted {stats['accepted_envs']}, "
        f"rejected {stats['rejected_seeds']})",
        f"gate passes: {counts(stats['gate_passes'])}",
        f"accepted envs: {stats['accepted_envs']} tasks: {stats['accepted_tasks']} "
        f"regression: {stats['regression_tasks']}",
        f"bands: {counts(stats['bands'])}",
        f"mutant_kill_rate: mean={num(stats['mutant_kill_rate_mean'])} min={num(stats['mutant_kill_rate_min'])}",
        f"cost: total=${stats['total_cost_usd']:.4f} per_task=${num(stats['cost_per_accepted_task_usd'])}",
        f"splits: {counts(stats['splits'])}",
        f"audit: false_accepts={stats['audit']['false_accepts']} n_audited={stats['audit']['n_audited']}",
    ]
