from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from envforge.procedural.gates import GateConfig, probe_pairs, run_gates, validate_module
from envforge.sandbox import SandboxSpec
from envforge.sandbox.process import ProcessBackend

FIXTURES = Path(__file__).parent / "fixtures"
SPEC = SandboxSpec(image="python:3.13-slim", cpu=1.0, memory_mb=1024, pids=256)


def _src(name: str) -> str:
    return (FIXTURES / f"modsum_{name}.py").read_text()


def _cfg(root: Path, call_timeout_s: float = 5.0) -> GateConfig:
    return GateConfig(ProcessBackend(), SPEC, call_timeout_s, root / "evidence" / "validate", root)


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("good")
    return root, validate_module(_src("good"), _src("solver"), cfg=_cfg(root))


def test_probe_pairs() -> None:
    pairs = probe_pairs(2)
    assert pairs == [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]


def test_good_passes_all_gates(good: tuple[Path, dict]) -> None:
    _, gates = good
    assert list(gates) == ["G1", "G2", "G3", "G4", "G5"]
    assert all(g.passed for g in gates.values()), {k: g.detail for k, g in gates.items()}
    assert gates["G1"].metrics == {"probes": 30, "nondeterministic": 0}
    assert gates["G2"].metrics["independent_agreement"] >= 0.95
    assert gates["G4"].metrics["false_accepts"] == 0
    assert gates["G5"].metrics["mutants"] >= 10
    assert gates["G5"].metrics["kill_rate"] >= 0.9


def test_evidence_files_written_with_relative_paths(good: tuple[Path, dict]) -> None:
    root, gates = good
    for name, g in gates.items():
        assert g.evidence_path == f"evidence/validate/{name}.json"
        assert not Path(g.evidence_path).is_absolute()
        payload = json.loads((root / g.evidence_path).read_text())
        assert payload["gate"] == name and payload["passed"] is True


def test_nondet_fails_g1(tmp_path: Path) -> None:
    gates = validate_module(_src("nondet"), _src("solver"), cfg=_cfg(tmp_path))
    assert list(gates) == ["G1"]
    assert not gates["G1"].passed
    assert gates["G1"].metrics["nondeterministic"] > 0


def test_import_os_fails_g1(tmp_path: Path) -> None:
    gates = validate_module(_src("import_os"), _src("solver"), cfg=_cfg(tmp_path))
    assert list(gates) == ["G1"]
    assert not gates["G1"].passed
    assert "os" in gates["G1"].detail


def test_lenient_fails_g4(tmp_path: Path) -> None:
    gates = validate_module(_src("lenient"), _src("solver"), cfg=_cfg(tmp_path))
    assert list(gates) == ["G1", "G2", "G3", "G4"]
    assert not gates["G4"].passed
    assert gates["G4"].metrics["false_accepts"] > 0


def test_wrong_solve_fails_g2(tmp_path: Path) -> None:
    gates = validate_module(_src("wrong_solve"), _src("solver"), cfg=_cfg(tmp_path))
    assert list(gates) == ["G1", "G2"]
    assert not gates["G2"].passed


def test_hang_fails_g2_fast(tmp_path: Path) -> None:
    start = time.monotonic()
    gates = validate_module(_src("hang"), _src("solver"), cfg=_cfg(tmp_path, call_timeout_s=0.5))
    assert time.monotonic() - start < 60
    assert list(gates) == ["G1", "G2"]
    assert not gates["G2"].passed
    assert "timeout" in gates["G2"].detail


def test_bad_check_return_fails_g2(tmp_path: Path) -> None:
    gates = validate_module(_src("bad_check_return"), _src("solver"), cfg=_cfg(tmp_path))
    assert list(gates) == ["G1", "G2"]
    assert not gates["G2"].passed
    assert "bad_return" in gates["G2"].detail


def test_run_gates_stop_after_g1_without_solver(tmp_path: Path) -> None:
    run = run_gates(_src("good"), None, cfg=_cfg(tmp_path), stop_after="G1")
    assert list(run.gates) == ["G1"]
    assert run.gates["G1"].passed
    assert run.levels == 6
    assert [(p.seed, p.difficulty) for p in run.probes] == probe_pairs(6)
    assert all(set(p.problem) >= {"prompt", "answer", "params"} for p in run.probes)
    assert run.mutant_kill_rate is None
