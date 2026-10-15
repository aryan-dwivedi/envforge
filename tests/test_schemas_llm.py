from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from envforge.answer import FORMAT_INSTRUCTION, extract_answer, task_messages
from envforge.config import ConfigError, EmbedderConfig, RoleConfig
from envforge.llm import LLM, ChatResult, Embedder
from envforge.schemas import Budgets, GateResult, PassStats, RunObservation, StateRef, ValidationReport


def test_extract_answer() -> None:
    assert extract_answer("a <answer>1</answer> b <answer> 2 </answer>") == "2"
    assert extract_answer("no tags") is None
    assert extract_answer("<answer>\nx\ny\n</answer>") == "x\ny"


def test_task_messages() -> None:
    assert task_messages("q") == [
        {"role": "system", "content": FORMAT_INSTRUCTION},
        {"role": "user", "content": "q"},
    ]


def test_schemas_frozen_and_forbid() -> None:
    b = Budgets()
    with pytest.raises(ValidationError):
        b.max_turns = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Budgets(bogus=1)  # type: ignore[call-arg]
    assert StateRef(kind="generator", ref="r").params == {}


def _report() -> ValidationReport:
    return ValidationReport(
        task_id="t",
        verifier_id="v",
        pipeline_version="0",
        gates={"G1": GateResult(passed=True)},
        mutant_kill_rate=1.0,
        solver_panel={"m": PassStats(n=4, successes=2, k=4, pass_at_1=0.5, pass_at_k=0.9)},
        band="target",
        near_dup_max_sim=0.1,
        contamination={},
        cost_usd=0.0,
    )


def test_validation_report_mutable() -> None:
    r = _report()
    r.lifecycle.append(RunObservation(run_id="x", observed_at="now", n=1, pass_rate=1.0))
    assert len(r.lifecycle) == 1
    assert _report().lifecycle == []
    with pytest.raises(ValidationError):
        ValidationReport(**_report().model_dump(), extra_field=1)


def _role() -> RoleConfig:
    return RoleConfig(
        base_url="http://x",
        model="m1",
        api_key_env="ENVFORGE_TEST_KEY",
        price_in_per_mtok=2.0,
        price_out_per_mtok=10.0,
    )


def test_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVFORGE_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="^env var ENVFORGE_TEST_KEY not set for model m1$"):
        LLM(_role()).chat([{"role": "user", "content": "hi"}])
    emb = Embedder(EmbedderConfig(base_url="http://x", model="e1", api_key_env="ENVFORGE_TEST_KEY"))
    with pytest.raises(ConfigError, match="^env var ENVFORGE_TEST_KEY not set for model e1$"):
        emb.embed(["a"])


class _FakeCompletions:
    def __init__(self, resp: object) -> None:
        self.resp = resp
        self.kwargs: dict = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.resp


def _fake_client(resp: object) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(resp)))


def test_chat_cost_and_defaults() -> None:
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )
    llm = LLM(_role())
    llm._client = _fake_client(resp)
    out = llm.chat([{"role": "user", "content": "hi"}])
    assert out == ChatResult(text="hello", input_tokens=1000, output_tokens=500, cost_usd=pytest.approx(0.007))
    kw = llm._client.chat.completions.kwargs
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 8000 and kw["model"] == "m1"
    llm.chat([], temperature=0.0, max_tokens=5)
    kw = llm._client.chat.completions.kwargs
    assert kw["temperature"] == 0.0 and kw["max_tokens"] == 5


def test_chat_usage_none() -> None:
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)
    llm = LLM(_role())
    llm._client = _fake_client(resp)
    assert llm.chat([]) == ChatResult(text="", input_tokens=0, output_tokens=0, cost_usd=0.0)


def test_embed_order() -> None:
    data = [SimpleNamespace(index=1, embedding=[2.0]), SimpleNamespace(index=0, embedding=[1.0])]
    emb = Embedder(EmbedderConfig(base_url="http://x", model="e1", api_key_env="K"))
    calls: dict = {}

    def create(**kwargs: object) -> object:
        calls.update(kwargs)
        return SimpleNamespace(data=data)

    emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert emb.embed(["a", "b"]) == [[1.0], [2.0]]
    assert calls == {"model": "e1", "input": ["a", "b"]}
    assert emb.model == "e1"
