"""Export accepted tasks as a self-contained OpenEnv server package."""

from __future__ import annotations

import importlib.resources
import json
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from envforge.sandbox.harness import runner_source
from envforge.store import Store

_TEMPLATE_FILES = (
    "models.py",
    "README.md",
    "server/__init__.py",
    "server/environment.py",
    "server/app.py",
    "server/requirements.txt",
    "server/Dockerfile",
)


def _template() -> Traversable:
    return importlib.resources.files("envforge.export") / "openenv_template"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def export_openenv(
    store: Store, out: Path, *, split: str = "train", include_regression: bool = False
) -> dict[str, Any]:
    """Write the OpenEnv package for the accepted tasks of ``split`` into ``out``."""
    tasks = store.load_tasks(split, include_regression)
    if not tasks:
        raise ValueError(f"no accepted tasks in split {split!r}")
    out = Path(out)
    template = _template()
    for rel in _TEMPLATE_FILES:
        node = template
        for part in rel.split("/"):
            node = node / part
        _write(out / rel, node.read_bytes())
    server = out / "server"
    _write(server / "runner.py", runner_source())
    _write(server / "answer.py", (importlib.resources.files("envforge") / "answer.py").read_bytes())
    rows = []
    for t in tasks:
        difficulty = int((t.difficulty_params or {}).get("difficulty", 0))
        rows.append({"task_id": t.task_id, "env_id": t.env_id, "difficulty": difficulty})
    for env_id in sorted({t.env_id for t in tasks}):
        _write(server / "generators" / f"{env_id}.py", store.load_generator_src(env_id).encode())
    _write(server / "tasks.json", (json.dumps(rows, indent=2) + "\n").encode())
    return {"out": str(out), "tasks": len(rows)}
