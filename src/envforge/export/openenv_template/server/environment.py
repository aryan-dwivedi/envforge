"""Procedural single-step environment backed by bundled generator modules."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_SERVER = Path(__file__).resolve().parent
_ROOT = str(_SERVER.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import EnvforgeAction, EnvforgeObservation  # noqa: E402
from openenv.core.env_server import Environment, State  # noqa: E402

from server.answer import FORMAT_INSTRUCTION, extract_answer  # noqa: E402

SEED_BASE = 10_000
CALL_TIMEOUT_S = 10.0


def _load_tasks() -> list[dict[str, Any]]:
    return json.loads((_SERVER / "tasks.json").read_text(encoding="utf-8"))


def _run(module: str, fn: str, args: list[Any]) -> Any:
    """Run one call through ``runner.py`` in a subprocess; raise on failure."""
    request = json.dumps({"timeout_s": CALL_TIMEOUT_S, "calls": [{"module": module, "fn": fn, "args": args}]})
    proc = subprocess.run(
        [sys.executable, "-s", "-P", "runner.py"],
        input=request.encode(),
        capture_output=True,
        cwd=_SERVER,
        timeout=CALL_TIMEOUT_S,
    )
    result = json.loads(proc.stdout)["results"][0]
    if not result.get("ok"):
        raise RuntimeError(f"{fn} failed: {result.get('error')}: {result.get('message', '')}")
    return result["value"]


class EnvforgeProceduralEnv(Environment[EnvforgeAction, EnvforgeObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._tasks = _load_tasks()
        self._by_id = {t["task_id"]: t for t in self._tasks}
        self._task: dict[str, Any] | None = None
        self._problem: dict[str, Any] | None = None
        self._seed = 0
        self._state = State(episode_id=None, step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> EnvforgeObservation:
        s = seed or 0
        task = self._by_id[task_id] if task_id is not None else self._tasks[s % len(self._tasks)]
        self._seed = SEED_BASE + s
        self._problem = _run(f"generators/{task['env_id']}.py", "gen", [self._seed, task["difficulty"]])
        self._task = task
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()), step_count=0, task_id=task["task_id"], seed=self._seed
        )
        return EnvforgeObservation(
            prompt=FORMAT_INSTRUCTION + "\n\n" + self._problem["prompt"],
            task_id=task["task_id"],
            done=False,
            reward=None,
        )

    def step(self, action: EnvforgeAction, timeout_s: float | None = None, **kwargs: Any) -> EnvforgeObservation:
        if self._task is None or self._problem is None:
            return EnvforgeObservation(done=True, reward=0.0, feedback="call reset first")
        self._state.step_count += 1
        extracted = extract_answer(f"<answer>{action.answer}</answer>")
        reward = 0.0
        if extracted is not None:
            try:
                value = _run(f"generators/{self._task['env_id']}.py", "check", [self._problem, extracted])
                reward = 1.0 if float(value) == 1.0 else 0.0
            except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired):
                reward = 0.0
        return EnvforgeObservation(
            task_id=self._task["task_id"],
            done=True,
            reward=reward,
            feedback="correct" if reward == 1.0 else "incorrect",
        )

    @property
    def state(self) -> State:
        return self._state
