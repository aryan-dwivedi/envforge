from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from envforge.config import ConfigError, SuiteConfig
from envforge.llm import ChatResult
from envforge.procedural.dedup import (
    ContaminationIndex,
    DedupRecord,
    g10,
    minhash_from_values,
    minhash_of,
    recheck,
)
from envforge.procedural.profile import band, pass_at_k, profile
from envforge.sandbox import SandboxSpec
from envforge.sandbox.harness import prepare_base
from envforge.sandbox.process import ProcessBackend
from envforge.schemas import PassStats

FIXTURE = Path(__file__).parent / "fixtures" / "modsum_good.py"
SPEC = SandboxSpec(image="python:3.13-slim", cpu=1.0, memory_mb=512, pids=64)


def ps(c: int, n: int = 100) -> PassStats:
    return PassStats(n=n, successes=c, k=8, pass_at_1=c / n, pass_at_k=pass_at_k(n, c, 8))


def test_band_boundaries() -> None:
    assert band([ps(90), ps(0)], ps(1)) == "target"
    assert band([ps(91)], ps(1)) == "too_easy"
    assert band([ps(0), ps(0)], ps(0)) == "unsolved"
    assert band([ps(5)], ps(0)) == "too_hard"


def test_pass_at_k() -> None:
    assert pass_at_k(8, 0, 8) == 0
    assert pass_at_k(8, 1, 8) == 1
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)


class FixedEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self.vec = vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vec) for _ in texts]


PROMPT = "compute the sum of these seventeen numbers modulo a small prime and report only the final value now"


def _run(vec: list[float], accepted: list[DedupRecord], index: ContaminationIndex):
    return g10(
        description="Some description of a task family.",
        dedup_prompts=["p one", "p two"],
        probe_prompts=[PROMPT],
        embedder=FixedEmbedder(vec),
        accepted=accepted,
        index=index,
    )


def test_g10_embedding_identical_fails_orthogonal_passes() -> None:
    empty = ContaminationIndex.from_suites([])
    mh = [int(v) for v in minhash_of("totally unrelated words here ok").hashvalues]
    other = DedupRecord("env_other", [1.0, 0.0], mh)
    same = _run([2.0, 0.0], [other], empty)
    assert not same.gate.passed and "env_other" in same.gate.detail
    assert same.near_dup_max_sim == pytest.approx(1.0)
    ortho = _run([0.0, 3.0], [other], empty)
    assert ortho.gate.passed and ortho.contamination == {}
    assert "no eval suites registered" in ortho.gate.detail
    assert set(ortho.gate.metrics) == {"near_dup_max_sim", "max_jaccard", "contamination_max"}
    # same text -> jaccard 1.0 caught by recheck even with orthogonal embedding
    r = recheck([1.0, 0.0], ortho.minhash, [DedupRecord("env_twin", [0.0, 1.0], ortho.minhash)])
    assert not r.passed and "env_twin" in r.detail


def test_contamination(tmp_path: Path) -> None:
    p = tmp_path / "suite.jsonl"
    p.write_text(json.dumps({"q": "x " + PROMPT.upper() + " y"}) + "\n" + json.dumps({"q": "short"}) + "\n")
    index = ContaminationIndex.from_suites([SuiteConfig(name="evalx", path=p, field="q")])
    res = _run([1.0, 0.0], [], index)
    assert res.contamination["evalx"] > 0
    assert not res.gate.passed and "evalx" in res.gate.detail
    assert index.rates(["too few tokens"]) == {"evalx": 0.0}


def test_suite_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ContaminationIndex.from_suites([SuiteConfig(name="m", path=tmp_path / "nope.jsonl", field="q")])
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"other": "x"}\n')
    with pytest.raises(ConfigError):
        ContaminationIndex.from_suites([SuiteConfig(name="b", path=bad, field="q")])


def test_minhash_round_trip() -> None:
    m = minhash_of(PROMPT)
    rebuilt = minhash_from_values([int(v) for v in m.hashvalues])
    assert m.jaccard(rebuilt) == 1.0


def _load_solve():
    spec = importlib.util.spec_from_file_location("modsum_good_fixture", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


class Correct:
    model = "correct"

    def __init__(self) -> None:
        self.solve = _load_solve()

    def chat(self, messages, *, temperature=None, max_tokens=None) -> ChatResult:
        assert temperature == 1.0
        return ChatResult(f"<answer>{self.solve(messages[-1]['content'])}</answer>", 10, 2, 0.01)


class Wrong:
    model = "wrong"

    def chat(self, messages, *, temperature=None, max_tokens=None) -> ChatResult:
        return ChatResult("<answer>-1</answer>", 10, 2, 0.001)


class Raises:
    model = "raises"

    def chat(self, messages, *, temperature=None, max_tokens=None) -> ChatResult:
        raise RuntimeError("api down")


@pytest.fixture
def base():
    sb = prepare_base(ProcessBackend(), SPEC, {"generator.py": FIXTURE.read_bytes()})
    yield sb
    sb.destroy()


def test_profile_too_easy_and_errors(base) -> None:
    res = profile(base, levels=2, panel=[("fa", Correct()), ("fb", Raises())], policy=Wrong(), call_timeout_s=10)
    assert res.llm_errors == 16
    assert res.cost_usd == pytest.approx(16 * 0.01 + 16 * 0.001)
    assert [lv.band for lv in res.levels] == ["too_easy", "too_easy"]
    lv = res.levels[0]
    assert set(lv.panel) == {"fa/correct", "fb/raises"} and lv.policy_key == "policy/wrong"
    assert lv.panel["fa/correct"].successes == 8 and lv.panel["fb/raises"].successes == 0
    assert not lv.gate.passed and lv.gate.metrics["llm_errors"] == 16


def test_profile_target_and_unsolved(base) -> None:
    res = profile(base, levels=1, panel=[("fb", Wrong())], policy=Correct(), call_timeout_s=10, concurrency=2)
    lv = res.levels[0]
    assert lv.band == "target" and lv.gate.passed
    assert lv.policy.pass_at_k == 1.0 and lv.gate.metrics["max_panel_pass1"] == 0.0
    res = profile(base, levels=1, panel=[("fb", Wrong())], policy=Raises(), call_timeout_s=10)
    assert res.levels[0].band == "unsolved" and res.llm_errors == 8
