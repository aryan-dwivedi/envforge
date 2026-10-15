"""OpenEnv export: self-contained package, environment semantics, HTTP smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from envforge.export.openenv import export_openenv


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported(accepted_store, tmp_path_factory):
    store, _ = accepted_store
    split = store.load_tasks()[0].split
    out = tmp_path_factory.mktemp("oe")
    result = export_openenv(store, out, split=split)
    assert result["tasks"] == len(store.load_tasks(split))
    saved = list(sys.path)
    mods = {k for k in sys.modules if k == "models" or k == "server" or k.startswith("server.")}
    sys.path.insert(0, str(out))
    try:
        yield out, _load(out / "server" / "environment.py", "server.environment")
    finally:
        sys.path[:] = saved
        for k in [k for k in sys.modules if k == "models" or k == "server" or k.startswith("server.")]:
            if k not in mods:
                sys.modules.pop(k)


def _answer(out: Path, seed: int, task: dict) -> str:
    gen = _load(out / "server" / "generators" / f"{task['env_id']}.py", "oe_gen")
    return gen.gen(10_000 + seed, task["difficulty"])["answer"]


def test_no_envforge_imports(exported):
    out, _ = exported
    for py in out.rglob("*.py"):
        text = py.read_text()
        assert "import envforge" not in text and "from envforge" not in text, py


def test_reset_step_correct_and_wrong(exported):
    out, env_mod = exported
    models = sys.modules["models"]
    tasks = json.loads((out / "server" / "tasks.json").read_text())
    task = tasks[3 % len(tasks)]
    env = env_mod.EnvforgeProceduralEnv()
    obs = env.reset(seed=3)
    assert obs.prompt and obs.done is False and obs.task_id == task["task_id"]
    ok = env.step(models.EnvforgeAction(answer=_answer(out, 3, task)))
    assert ok.reward == 1.0 and ok.done is True and ok.feedback == "correct"
    assert env.state.seed == 10_003 and env.state.step_count == 1
    env.reset(seed=3)
    bad = env.step(models.EnvforgeAction(answer="nope"))
    assert bad.reward == 0.0 and bad.done is True and bad.feedback == "incorrect"


def test_step_before_reset(exported):
    _, env_mod = exported
    obs = env_mod.EnvforgeProceduralEnv().step(sys.modules["models"].EnvforgeAction(answer="1"))
    assert obs.done is True and obs.reward == 0.0 and obs.feedback == "call reset first"


def test_reset_by_task_id(exported):
    out, env_mod = exported
    tasks = json.loads((out / "server" / "tasks.json").read_text())
    target = tasks[-1]["task_id"]
    assert env_mod.EnvforgeProceduralEnv().reset(seed=0, task_id=target).task_id == target


def test_http_reset(exported):
    from fastapi.testclient import TestClient

    out, _ = exported
    app = _load(out / "server" / "app.py", "server.app").app
    resp = TestClient(app).post("/reset", json={"seed": 3})
    assert resp.status_code == 200
    assert resp.json()["observation"]["prompt"]
