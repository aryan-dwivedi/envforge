"""Shared test doubles and the end-to-end pipeline store fixture."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from envforge.config import Config
from envforge.hashing import sha256_hex
from envforge.llm import ChatResult
from envforge.procedural.dedup import ContaminationIndex
from envforge.procedural.pipeline import Models, run_seed
from envforge.procedural.seeds import Seed
from envforge.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM:
    """ChatModel double: a callable reply, or a list consumed in order whose last item repeats."""

    def __init__(self, model: str, reply: Callable[[list[dict]], str] | list[str], cost_usd: float = 0.0) -> None:
        self.model = model
        self.reply = reply
        self.cost_usd = cost_usd
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()

    def chat(
        self, messages: list[dict], *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatResult:
        with self._lock:
            self.calls.append(messages)
            if callable(self.reply):
                text = None
            else:
                text = self.reply[min(len(self.calls), len(self.reply)) - 1]
        if text is None:
            text = self.reply(messages)  # type: ignore[operator]
        return ChatResult(text=text, input_tokens=0, output_tokens=0, cost_usd=self.cost_usd)


class ScriptedEmbedder:
    """EmbedModel double returning the same vector for every text."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]


def fenced(path: str | Path) -> str:
    """Return a fixture file wrapped in a python code fence."""
    p = Path(path)
    if not p.is_absolute():
        p = FIXTURES / p
    return f"Here is the module.\n\n```python\n{p.read_text()}```\n"


def _good_solve() -> Callable[[str], str]:
    spec = importlib.util.spec_from_file_location("modsum_good_reference", FIXTURES / "modsum_good.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def answer_rule(modulus: int | None) -> Callable[[list[dict]], str]:
    """Deterministic answerer: correct iff sha256(prompt) % modulus == 0; None means never."""
    solve = _good_solve()

    def reply(messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        if modulus is not None and int(sha256_hex(prompt), 16) % modulus == 0:
            return f"Working... <answer>{solve(prompt)}</answer>"
        return "<answer>-1</answer>"

    return reply


def make_config(tmp_path: Path, **overrides: object) -> Config:
    """Process-backend config built in memory; top-level keys may be overridden."""
    role = {"base_url": "https://llm.example.com/v1", "api_key_env": "ENVFORGE_TEST_KEY"}
    raw: dict[str, object] = {
        "store_dir": str(tmp_path / "store"),
        "sandbox": {"backend": "process", "cpu": 1.0, "memory_mb": 1024, "pids": 256},
        "limits": {"call_timeout_s": 5.0},
        "run": {"llm_concurrency": 4, "max_repairs": 3},
        "roles": {
            "builder": {**role, "model": "builder-model"},
            "repairer": {**role, "model": "builder-model"},
            "independent_solver": {**role, "model": "solver-model"},
            "panel": [{**role, "model": "panel-a", "family": "fam-a"}, {**role, "model": "panel-b", "family": "fam-b"}],
            "policy": {**role, "model": "policy-model"},
            "embedder": {**role, "model": "embed-model"},
        },
        "contamination": {"suites": []},
    }
    raw.update(overrides)
    return Config.model_validate(raw)


def make_models(
    builder: list[str],
    repairer: list[str],
    *,
    solver: list[str] | None = None,
    vector: list[float] | None = None,
    cost_usd: float = 0.0,
) -> Models:
    return Models(
        builder=ScriptedLLM("builder-model", builder, cost_usd),
        repairer=ScriptedLLM("builder-model", repairer, cost_usd),
        independent_solver=ScriptedLLM("solver-model", solver or [fenced("modsum_solver.py")], cost_usd),
        policy=ScriptedLLM("policy-model", answer_rule(3), cost_usd),
        panel=[
            ("fam-a", ScriptedLLM("panel-a", answer_rule(2), cost_usd)),
            ("fam-b", ScriptedLLM("panel-b", answer_rule(None), cost_usd)),
        ],
        embedder=ScriptedEmbedder(vector or [1.0, 0.0, 0.0]),
        contamination=ContaminationIndex({}),
    )


def make_seed(seed_id: str) -> Seed:
    return Seed(
        id=seed_id,
        title="Modular sum",
        description="Compute the sum of integers modulo a small modulus.",
        tags=("arithmetic",),
        license="CC0-1.0",
        source="envforge-builtin",
    )


E2E_SEED = "modular-sum"


@pytest.fixture(scope="session")
def accepted_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Store, Config]:
    """Run the e2e pipeline once: wrong builder module, repaired to the good one, accepted."""
    tmp = tmp_path_factory.mktemp("e2e")
    cfg = make_config(tmp)
    store = Store(cfg.store_dir)
    models = make_models([fenced("modsum_wrong_solve.py")], [fenced("modsum_good.py")], cost_usd=0.001)
    outcome = run_seed(make_seed(E2E_SEED), cfg, store, models)
    assert isinstance(outcome, list), outcome
    return store, cfg
