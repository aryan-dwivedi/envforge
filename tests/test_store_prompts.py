from __future__ import annotations

from pathlib import Path

import pytest

from envforge.procedural.contract import CONTRACT_TEXT, IMPORT_ALLOWLIST
from envforge.procedural.seeds import Seed
from envforge.prompts import (
    prompts_hash,
    render_builder,
    render_independent_solver,
    render_repair,
)
from envforge.schemas import (
    Budgets,
    Environment,
    GateResult,
    PassStats,
    Provenance,
    ReferenceSolution,
    RejectedRun,
    StateRef,
    Task,
    ValidationReport,
    Verifier,
)
from envforge.store import Store


def _env(env_id: str = "env_a") -> Environment:
    return Environment(
        env_id=env_id,
        family="procedural",
        image="process:python3.12",
        isolation_tier="process",
        tools=[],
        state_init=StateRef(kind="generator", ref="generator.py", params={"difficulty_levels": 6}),
        network_policy="none",
        harness_profiles=["single_turn_answer_tag"],
        provenance=Provenance(
            seed_id="s1",
            seed_source="builtin",
            license="MIT",
            generator_model="m",
            prompts_hash="h",
            pipeline_version="procedural@0.1.0",
            created_at="2026-01-01T00:00:00Z",
        ),
    )


def _verifier() -> Verifier:
    return Verifier(
        verifier_id="ver_a",
        kind="value_match",
        code_ref="generator.py::check@sha256:x",
        hidden_checks=[],
        rubric=None,
        deterministic=True,
        partial_credit=False,
    )


def _task(task_id: str, split: str = "train") -> Task:
    return Task(
        task_id=task_id,
        env_id="env_a",
        prompt="q",
        difficulty_params={"difficulty": 1},
        reference=ReferenceSolution(kind="program", code_ref="generator.py", entrypoint="solve"),
        verifier_id="ver_a",
        budgets=Budgets(),
        tags=["procedural"],
        split=split,  # type: ignore[arg-type]
    )


def _report(task_id: str) -> ValidationReport:
    return ValidationReport(
        task_id=task_id,
        verifier_id="ver_a",
        pipeline_version="0",
        gates={"G1": GateResult(passed=True, metrics={"x": 1.0})},
        mutant_kill_rate=1.0,
        solver_panel={"m": PassStats(n=8, successes=2, k=8, pass_at_1=0.25, pass_at_k=0.9)},
        band="target",
        near_dup_max_sim=0.1,
        contamination={},
        cost_usd=0.0,
    )


def _rejected(reason: str) -> RejectedRun:
    return RejectedRun(
        seed_id="s1",
        reason=reason,
        attempts=[{"G1": GateResult(passed=False, detail="d")}],
        cost_usd=0.5,
        created_at="now",
    )


def test_roundtrip(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.save_env(_env(), "GEN", "SOL", embedding=[0.5, 0.5], minhash=[1, 2, 3])
    s.save_verifier(_verifier())
    s.save_task(_task("t1"))
    s.save_report(_report("t1"))
    s.save_rejected(_rejected("r"))
    assert s.load_env("env_a") == _env()
    assert s.load_generator_src("env_a") == "GEN"
    assert (tmp_path / "environments/env_a/solver.py").read_text() == "SOL"
    assert s.load_verifier("ver_a") == _verifier()
    assert s.load_task("t1") == _task("t1")
    assert s.load_report("t1") == _report("t1")
    assert s.load_reports() == [_report("t1")]
    assert s.load_rejected() == [_rejected("r")]
    assert s.env_ids() == ["env_a"]
    assert s.accepted_dedup() == [("env_a", [0.5, 0.5], [1, 2, 3])]
    assert s.evidence_dir("s1/attempt1") == tmp_path / "evidence/s1/attempt1"
    assert s.evidence_dir("s1/attempt1").is_dir()
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_empty_store(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.load_tasks() == [] and s.load_reports() == [] and s.load_rejected() == []
    assert s.env_ids() == [] and s.accepted_dedup() == []


def test_load_tasks_split_and_regression(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.save_task(_task("t2", "eval"))
    s.save_task(_task("t1", "train"))
    s.save_task(_task("t0", "train"), regression=True)
    assert [t.task_id for t in s.load_tasks()] == ["t1", "t2"]
    assert [t.task_id for t in s.load_tasks("train")] == ["t1"]
    assert [t.task_id for t in s.load_tasks("train", include_regression=True)] == ["t0", "t1"]
    assert [t.task_id for t in s.load_tasks(include_regression=True)] == ["t0", "t1", "t2"]
    assert s.load_task("t0").task_id == "t0"
    with pytest.raises(FileNotFoundError):
        s.load_task("missing")


def test_accepted_dedup_multiple(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.save_env(_env("env_b"), "g", "s", embedding=[1.0], minhash=[9])
    s.save_env(_env("env_a"), "g", "s", embedding=[0.0], minhash=[8])
    assert s.accepted_dedup() == [("env_a", [0.0], [8]), ("env_b", [1.0], [9])]


def test_bad_ids(tmp_path: Path) -> None:
    s = Store(tmp_path)
    for bad in ("a/b", "..", "x..y", ""):
        with pytest.raises(ValueError):
            s.load_env(bad)
    with pytest.raises(ValueError):
        s.save_task(_task("../t"))
    with pytest.raises(ValueError):
        s.evidence_dir("../x")


def test_rejected_overwrite(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.save_rejected(_rejected("first"))
    s.save_rejected(_rejected("second"))
    assert [r.reason for r in s.load_rejected()] == ["second"]


def test_render_builder() -> None:
    seed = Seed(id="s1", title="Modular sums", description="Sum numbers mod m.", tags=(), license="MIT", source="x")
    text = render_builder(seed)
    assert "Modular sums" in text and "Sum numbers mod m." in text
    assert CONTRACT_TEXT.strip() in text
    assert ", ".join(sorted(IMPORT_ALLOWLIST)) in text
    assert "`solve` must derive the answer from the prompt text alone" in text
    assert "difficulty must monotonically increase problem size/search space" in text
    assert text.rstrip().endswith("```python fenced block containing the full module.")


def test_render_repair_truncates() -> None:
    text = render_repair("SRC_MODULE", "G2", "a" * 3990 + "b" * 100)
    assert "SRC_MODULE" in text and "G2" in text
    assert "a" * 3990 + "b" * 10 in text and "b" * 11 not in text
    assert "decide whether gen, solve or check is wrong" in text


def test_render_independent_solver() -> None:
    ex = [("prompt one", "1"), ("prompt two", "22"), ("prompt three", "333")]
    text = render_independent_solver("DESC", ex)
    assert "DESC" in text
    for p, a in ex:
        assert p in text and a in text
    assert "def gen" not in text
    assert "write `solve(prompt: str) -> str` using only the allowlist" in text
    with pytest.raises(ValueError):
        render_independent_solver("DESC", ex[:2])


def test_prompts_hash() -> None:
    h = prompts_hash()
    assert h == prompts_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
