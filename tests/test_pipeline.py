from __future__ import annotations

from pathlib import Path

import pytest
from conftest import E2E_SEED, fenced, make_config, make_models, make_seed

from envforge.config import Config, ConfigError
from envforge.llm import Embedder
from envforge.procedural.pipeline import build_models, run_seed
from envforge.schemas import RejectedRun
from envforge.store import Store


def test_e2e_accepts_after_repair(accepted_store: tuple[Store, Config]) -> None:
    store, _ = accepted_store
    tasks = store.load_tasks()
    assert len(tasks) >= 1
    reports = store.load_reports()
    assert reports
    for r in reports:
        assert r.gates["G2"].passed
        assert r.mutant_kill_rate >= 0.9
        assert {"G1", "G2", "G3", "G4", "G5", "G9", "G10"} <= set(r.gates)
        assert r.cost_usd > 0
    assert all(r.band == "target" for r in reports if r.task_id in {t.task_id for t in tasks})
    assert not (store.root / "rejected" / f"{E2E_SEED}.json").exists()
    env = store.load_env(tasks[0].env_id)
    assert env.provenance.generator_model == "builder-model"
    assert env.isolation_tier == "process"
    assert (store.root / "evidence" / E2E_SEED / "attempt2" / "G5.json").is_file()


def test_identical_module_rejected_by_g10(accepted_store: tuple[Store, Config]) -> None:
    store, cfg = accepted_store
    models = make_models([fenced("modsum_good.py")], [fenced("modsum_good.py")])
    outcome = run_seed(make_seed("modular-sum-dup"), cfg, store, models)
    assert isinstance(outcome, RejectedRun)
    assert outcome.reason.startswith("G10")
    assert store.load_rejected()[-1].seed_id == "modular-sum-dup"


def test_rerun_of_same_seed_is_idempotent(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, run={"llm_concurrency": 4, "max_repairs": 0})
    store = Store(cfg.store_dir)
    seed = make_seed("rerun")
    first = run_seed(seed, cfg, store, make_models([fenced("modsum_wrong_solve.py")], ["unused"]))
    assert isinstance(first, RejectedRun)

    good = make_models([fenced("modsum_good.py")], ["unused"])
    second = run_seed(seed, cfg, store, good)
    third = run_seed(seed, cfg, store, good)
    assert isinstance(second, list) and second
    assert third == second
    assert store.load_rejected() == []
    assert len(store.env_ids()) == 1


def test_exhausted_repairs_rejected(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, limits={"call_timeout_s": 2.0})
    store = Store(cfg.store_dir)
    models = make_models([fenced("modsum_wrong_solve.py")], [fenced("modsum_wrong_solve.py")])
    outcome = run_seed(make_seed("always-wrong"), cfg, store, models)
    assert isinstance(outcome, RejectedRun)
    assert outcome.reason == "gate G2 failed after 4 attempts"
    assert len(outcome.attempts) == 4
    assert all(not a["G2"].passed for a in outcome.attempts)
    assert store.load_rejected()[0].reason == outcome.reason


def test_no_python_block_is_g1_failure(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, run={"llm_concurrency": 4, "max_repairs": 0})
    store = Store(cfg.store_dir)
    models = make_models(["I cannot write that."], ["unused"])
    outcome = run_seed(make_seed("no-block"), cfg, store, models)
    assert isinstance(outcome, RejectedRun)
    assert outcome.reason == "gate G1 failed after 1 attempts"
    assert outcome.attempts[0]["G1"].detail == "no python block"


def test_build_models_embedder_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: Embedder, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(Embedder, "embed", boom)
    with pytest.raises(ConfigError, match=r"^embedder endpoint failed: connection refused$"):
        build_models(make_config(tmp_path))
