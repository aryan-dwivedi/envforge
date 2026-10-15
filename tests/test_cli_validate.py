from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from envforge.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _validate(generator: str) -> tuple[int, list[str]]:
    result = runner.invoke(
        app, ["validate", str(FIXTURES / f"modsum_{generator}.py"), "--solver", str(FIXTURES / "modsum_solver.py")]
    )
    return result.exit_code, result.output.splitlines()


def test_validate_good_exits_zero() -> None:
    code, lines = _validate("good")
    assert code == 0, lines
    for gate in ("G1", "G2", "G3", "G4", "G5"):
        assert any(line.startswith(f"{gate} PASS") for line in lines), lines
    assert "G1 PASS probes=30 nondeterministic=0" in lines


def test_validate_lenient_fails_g4() -> None:
    code, lines = _validate("lenient")
    assert code == 1
    assert any(line.startswith("G4 FAIL false_accepts=") for line in lines), lines
