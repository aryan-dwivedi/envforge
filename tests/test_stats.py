from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from envforge.cli import app
from envforge.config import Config
from envforge.schemas import GateResult, RejectedRun
from envforge.stats import compute_stats, format_stats
from envforge.store import Store


def test_stats_over_store(accepted_store: tuple[Store, Config], tmp_path: Path) -> None:
    store, _ = accepted_store
    snapshot = compute_stats(store)
    planted = Store(tmp_path / "copy")
    # Copy the accepted store into a fresh root so other tests see it unchanged.
    for p in store.root.rglob("*.json"):
        if "evidence" in p.parts:
            continue
        dest = planted.root / p.relative_to(store.root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(p.read_text())
    planted.save_rejected(
        RejectedRun(
            seed_id="planted",
            reason="gate G2 failed after 1 attempts",
            attempts=[{"G1": GateResult(passed=True), "G2": GateResult(passed=False)}],
            cost_usd=0.5,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    s = compute_stats(planted)
    n_tasks = len(store.load_tasks())
    assert s["accepted_envs"] == 1
    assert s["accepted_tasks"] == n_tasks
    assert s["rejected_seeds"] == snapshot["rejected_seeds"] + 1
    assert s["seeds_attempted"] == 1 + s["rejected_seeds"]
    gp, before = s["gate_passes"], snapshot["gate_passes"]
    assert gp["G1"] == before["G1"] + 1 and gp["G2"] == before["G2"]
    assert gp["G2"] >= 1 and gp["G5"] >= 1 and gp["G10"] >= 1 and gp["G9"] >= 1
    assert s["mutant_kill_rate_min"] >= 0.9
    assert abs(s["total_cost_usd"] - (snapshot["total_cost_usd"] + 0.5)) < 1e-9
    assert abs(s["cost_per_accepted_task_usd"] - s["total_cost_usd"] / n_tasks) < 1e-9
    assert sum(s["bands"].values()) == s["accepted_tasks"] + s["regression_tasks"]
    assert sum(s["splits"].values()) == n_tasks
    assert s["audit"] == {"false_accepts": 0, "n_audited": 0}
    assert any(line.startswith("gate passes: G1:") for line in format_stats(s))


def test_run_requires_seed_selection(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["run", "--config", str(tmp_path / "envforge.toml")])
    assert result.exit_code == 2
    assert "pass --seeds or --all" in result.output
