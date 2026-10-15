"""Filesystem store for accepted environments, tasks, reports and rejections."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from pydantic import BaseModel

from envforge.schemas import Environment, RejectedRun, Task, ValidationReport, Verifier


def _check_id(value: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"unsafe id: {value!r}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(path, model.model_dump_json(indent=2))


class Store:
    """Directory layout rooted at `root`; all writes are atomic."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.lock = threading.Lock()

    # paths

    def _env_dir(self, env_id: str) -> Path:
        return self.root / "environments" / _check_id(env_id)

    def evidence_dir(self, run_key: str) -> Path:
        """`run_key` is `f"{seed_id}/attempt{n}"`; each path segment is checked."""
        parts = run_key.split("/")
        for p in parts:
            _check_id(p)
        d = self.root.joinpath("evidence", *parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # saves

    def save_env(
        self,
        env: Environment,
        generator_src: str,
        solver_src: str,
        *,
        embedding: list[float],
        minhash: list[int],
    ) -> None:
        d = self._env_dir(env.env_id)
        _atomic_write(d / "generator.py", generator_src)
        _atomic_write(d / "solver.py", solver_src)
        _atomic_write(
            d / "dedup.json",
            json.dumps({"embedding": list(embedding), "minhash": [int(x) for x in minhash]}, indent=2),
        )
        _write_model(d / "env.json", env)

    def save_verifier(self, v: Verifier) -> None:
        _write_model(self.root / "verifiers" / f"{_check_id(v.verifier_id)}.json", v)

    def save_task(self, t: Task, *, regression: bool = False) -> None:
        sub = "regression" if regression else "tasks"
        _write_model(self.root / sub / f"{_check_id(t.task_id)}.json", t)

    def save_report(self, r: ValidationReport) -> None:
        _write_model(self.root / "reports" / f"{_check_id(r.task_id)}.json", r)

    def save_rejected(self, r: RejectedRun) -> None:
        _write_model(self.root / "rejected" / f"{_check_id(r.seed_id)}.json", r)

    def clear_rejected(self, seed_id: str) -> None:
        """Drop a stale rejection once the seed is accepted on a rerun."""
        (self.root / "rejected" / f"{_check_id(seed_id)}.json").unlink(missing_ok=True)

    # loads

    def load_env(self, env_id: str) -> Environment:
        return Environment.model_validate_json((self._env_dir(env_id) / "env.json").read_text(encoding="utf-8"))

    def load_generator_src(self, env_id: str) -> str:
        return (self._env_dir(env_id) / "generator.py").read_text(encoding="utf-8")

    def load_verifier(self, verifier_id: str) -> Verifier:
        p = self.root / "verifiers" / f"{_check_id(verifier_id)}.json"
        return Verifier.model_validate_json(p.read_text(encoding="utf-8"))

    def load_task(self, task_id: str) -> Task:
        name = f"{_check_id(task_id)}.json"
        for sub in ("tasks", "regression"):
            p = self.root / sub / name
            if p.is_file():
                return Task.model_validate_json(p.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"task not found: {task_id}")

    def load_tasks(self, split: str | None = None, include_regression: bool = False) -> list[Task]:
        subs = ["tasks", "regression"] if include_regression else ["tasks"]
        out: list[Task] = []
        for sub in subs:
            d = self.root / sub
            if not d.is_dir():
                continue
            for p in d.glob("*.json"):
                t = Task.model_validate_json(p.read_text(encoding="utf-8"))
                if split is None or t.split == split:
                    out.append(t)
        return sorted(out, key=lambda t: t.task_id)

    def load_report(self, task_id: str) -> ValidationReport:
        p = self.root / "reports" / f"{_check_id(task_id)}.json"
        return ValidationReport.model_validate_json(p.read_text(encoding="utf-8"))

    def load_reports(self) -> list[ValidationReport]:
        d = self.root / "reports"
        if not d.is_dir():
            return []
        return [ValidationReport.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]

    def load_rejected(self) -> list[RejectedRun]:
        d = self.root / "rejected"
        if not d.is_dir():
            return []
        return [RejectedRun.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]

    def env_ids(self) -> list[str]:
        d = self.root / "environments"
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if (p / "env.json").is_file())

    def accepted_dedup(self) -> list[tuple[str, list[float], list[int]]]:
        d = self.root / "environments"
        if not d.is_dir():
            return []
        out: list[tuple[str, list[float], list[int]]] = []
        for p in sorted(d.iterdir()):
            f = p / "dedup.json"
            if f.is_file():
                data = json.loads(f.read_text(encoding="utf-8"))
                out.append((p.name, [float(x) for x in data["embedding"]], [int(x) for x in data["minhash"]]))
        return out
