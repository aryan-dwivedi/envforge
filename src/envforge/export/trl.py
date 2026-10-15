"""TRL dataset export and a GRPO-compatible reward function.

Rows carry the full problem (including the hidden answer) in a ``problem``
column; TRL passes extra columns only to reward functions, never to the model.
"""

from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from envforge import __version__
from envforge.answer import task_messages
from envforge.hashing import canonical_json
from envforge.procedural import scoring
from envforge.sandbox import Sandbox, SandboxSpec
from envforge.sandbox.harness import prepare_base
from envforge.sandbox.process import ProcessBackend
from envforge.schemas import Task
from envforge.store import Store

EXPORT_SEED_BASE = 10_000
MODULE = "generator.py"
_SPEC = SandboxSpec(image="python:3.13-slim", cpu=1.0, memory_mb=1024, pids=256)


def env_sandbox(store: Store, env_id: str) -> Sandbox:
    """Process sandbox holding ``runner.py`` and the env's ``generator.py``."""
    return prepare_base(ProcessBackend(), _SPEC, {MODULE: store.load_generator_src(env_id).encode()})


def task_difficulty(task: Task) -> int:
    return int((task.difficulty_params or {})["difficulty"])


def generate_for_env(
    store: Store, env_id: str, pairs: list[tuple[int, int]], *, call_timeout_s: float
) -> list[scoring.Generated]:
    """Generate ``(seed, difficulty)`` pairs on a fork of a fresh env sandbox."""
    base = env_sandbox(store, env_id)
    try:
        fork = base.fork()
        try:
            return scoring.generate(fork, MODULE, pairs, call_timeout_s=call_timeout_s)
        finally:
            fork.destroy()
    finally:
        base.destroy()


def export_trl(
    store: Store,
    out: Path,
    *,
    split: str = "train",
    instances_per_task: int = 64,
    include_regression: bool = False,
    call_timeout_s: float = 5.0,
) -> dict:
    """Write ``out/data.jsonl`` and ``out/manifest.json``; return the manifest plus counts."""
    tasks = store.load_tasks(split, include_regression)
    by_env: dict[str, list[Task]] = {}
    for t in tasks:
        by_env.setdefault(t.env_id, []).append(t)
    rows: list[dict[str, Any]] = []
    failures = 0
    for env_id, env_tasks in by_env.items():
        keys: list[Task] = []
        pairs: list[tuple[int, int]] = []
        for t in env_tasks:
            d = task_difficulty(t)
            for i in range(instances_per_task):
                keys.append(t)
                pairs.append((EXPORT_SEED_BASE + i, d))
        for t, g in zip(keys, generate_for_env(store, env_id, pairs, call_timeout_s=call_timeout_s), strict=True):
            if g.problem is None:
                failures += 1
                continue
            rows.append(
                {
                    "prompt": task_messages(g.problem["prompt"]),
                    "task_id": t.task_id,
                    "env_id": env_id,
                    "seed": g.seed,
                    "problem": canonical_json(g.problem),
                }
            )
    rows.sort(key=lambda r: (r["task_id"], r["seed"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "data.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "store_dir": str(Path(store.root).resolve()),
        "split": split,
        "tasks": [t.task_id for t in tasks],
        "envforge_version": __version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "rows": len(rows), "gen_failures": failures}


def _destroy_all(cache: dict[str, Sandbox], lock: threading.Lock) -> None:
    with lock:
        for sb in cache.values():
            try:
                sb.destroy()
            except Exception:
                pass
        cache.clear()


def _text(c: Any) -> str:
    if isinstance(c, list):
        return str(c[-1]["content"]) if c else ""
    return str(c)


def make_reward_func(store_dir: str | Path, *, call_timeout_s: float = 5.0) -> Callable[..., list[float]]:
    """Return ``envforge_reward(prompts, completions, task_id, problem, **kwargs)`` for TRL."""
    store = Store(Path(store_dir))
    cache: dict[str, Sandbox] = {}
    env_of: dict[str, str] = {}
    lock = threading.Lock()

    def _sandbox(env_id: str) -> Sandbox:
        sb = cache.get(env_id)
        if sb is None:
            sb = cache[env_id] = env_sandbox(store, env_id)
        return sb

    def envforge_reward(
        prompts: list[Any], completions: list[Any], task_id: list[str], problem: list[str], **kwargs: Any
    ) -> list[float]:
        n = len(prompts)
        rewards = [0.0] * n
        groups: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
        for i in range(n):
            try:
                tid = task_id[i]
                with lock:
                    if tid not in env_of:
                        env_of[tid] = store.load_task(tid).env_id
                    env_id = env_of[tid]
                groups.setdefault(env_id, []).append((i, json.loads(problem[i]), _text(completions[i])))
            except Exception:
                continue
        for env_id, items in groups.items():
            try:
                with lock:
                    results = scoring.score(
                        _sandbox(env_id), MODULE, [(p, t) for _, p, t in items], call_timeout_s=call_timeout_s
                    )
            except Exception:
                continue
            for (i, _, _), s in zip(items, results, strict=True):
                rewards[i] = float(s.value) if s.error is None else 0.0
        return rewards

    envforge_reward.__name__ = "envforge_reward"
    finalizer = weakref.finalize(envforge_reward, _destroy_all, cache, lock)
    envforge_reward.close = finalizer  # type: ignore[attr-defined]
    return envforge_reward
