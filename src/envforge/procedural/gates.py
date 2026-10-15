"""Validation gates G1-G5 for procedural generator modules.

Every sandbox operation of a gate runs on a fork of the base sandbox, so
judging never mutates the base. Each gate writes its full evidence as JSON
and returns a :class:`GateResult` whose ``detail`` summarises at most the
first few failures.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from envforge.answer import extract_answer
from envforge.hashing import canonical_json, sha256_hex
from envforge.procedural.contract import static_check
from envforge.procedural.mutate import mutants
from envforge.procedural.scoring import Score, generate, score
from envforge.sandbox import Sandbox, SandboxBackend, SandboxError, SandboxSpec
from envforge.sandbox.harness import prepare_base, run_calls
from envforge.schemas import GateResult

GATES = ("G1", "G2", "G3", "G4", "G5")
PROBE_SEEDS = range(0, 5)
G1_HASH_SEEDS = ("0", "1", "2")
G2_REFERENCE_RUNS = 5
G2_MIN_AGREEMENT = 0.95
G3_HASH_SEEDS = ("0", "1", "2", "3", "4")
G5_MIN_MUTANTS = 10
G5_MIN_KILL_RATE = 0.90
MAX_DETAIL_FAILURES = 5
PROMPT_EXCERPT = 500
GENERATOR = "generator.py"
SOLVER = "solver.py"
# Metrics that are fractions; every other metric is a count.
RATE_METRICS = frozenset({"independent_agreement", "kill_rate"})


@dataclass(frozen=True)
class Probe:
    seed: int
    difficulty: int
    problem: dict[str, Any]


@dataclass(frozen=True)
class GateConfig:
    backend: SandboxBackend
    spec: SandboxSpec
    call_timeout_s: float
    evidence_dir: Path
    evidence_root: Path  # evidence_path is stored relative to this directory


@dataclass(frozen=True)
class GateRun:
    gates: dict[str, GateResult]
    probes: list[Probe] = field(default_factory=list)
    levels: int = 0
    mutant_kill_rate: float | None = None


def probe_pairs(levels: int) -> list[tuple[int, int]]:
    """Return the probe set: seeds 0..4 x difficulties 1..levels."""
    return [(s, d) for s in PROBE_SEEDS for d in range(1, levels + 1)]


def _difficulty_levels(source: str) -> int:
    """Read ``DIFFICULTY_LEVELS`` from a module that already passed ``static_check``."""
    value: int | None = None
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "DIFFICULTY_LEVELS" for t in targets):
            assert isinstance(stmt.value, ast.Constant)
            value = int(stmt.value.value)  # the last binding wins, as in static_check
    if value is None:
        raise ValueError("DIFFICULTY_LEVELS not found")
    return value


def _answer_tag(text: str) -> str:
    return f"<answer>{text}</answer>"


def _failure(
    probe: Probe | None,
    *,
    seed: int | None = None,
    difficulty: int | None = None,
    got: Any = None,
    check: float | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    problem = probe.problem if probe else {}
    entry = {
        "seed": probe.seed if probe else seed,
        "difficulty": probe.difficulty if probe else difficulty,
        "prompt": str(problem.get("prompt", ""))[:PROMPT_EXCERPT],
        "expected": problem.get("answer"),
        "got": got,
        "check": check,
        "error": error,
    }
    entry.update(extra)
    return entry


def _error_text(kind: str | None, message: str) -> str | None:
    if kind is None:
        return None
    return f"{kind}: {message[:PROMPT_EXCERPT]}" if message else kind


def _call_error(fn: str, result: dict[str, Any]) -> str:
    return f"{fn} {_error_text(result.get('error', 'runner_crash'), result.get('message', ''))}"


def _summarise(failures: list[dict[str, Any]], header: str = "") -> str:
    lines = [header] if header else []
    if failures:
        lines.append(f"{len(failures)} failure(s); first {min(len(failures), MAX_DETAIL_FAILURES)}:")
    for f in failures[:MAX_DETAIL_FAILURES]:
        head = " ".join(f"{k}={f[k]!r}" for k in f if k != "prompt")
        lines.append(f"- {head}\n  prompt: {f['prompt']!r}")
    return "\n".join(lines)


class Validator:
    """Runs G1-G5 against one generator module and an optional independent solver.

    Owns the base sandbox: ``generator.py``, ``solver.py`` when a solver is
    given, and ``mutants/m{i}.py`` once G5 runs. G1 creates it; later gates
    require a passed G1.
    """

    def __init__(self, cfg: GateConfig, generator_src: str, solver_src: str | None) -> None:
        self.cfg = cfg
        self.generator_src = generator_src
        self.solver_src = solver_src
        self.levels = 0
        self.probes: list[Probe] = []
        self.mutant_kill_rate: float | None = None
        self._base: Sandbox | None = None

    def __enter__(self) -> Validator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._base is not None:
            self._base.destroy()
            self._base = None

    # -- helpers ---------------------------------------------------------

    @contextmanager
    def _fork(self) -> Iterator[Sandbox]:
        if self._base is None or not self.probes:
            raise RuntimeError("G1 must pass before later gates run")
        fork = self._base.fork()
        try:
            yield fork
        finally:
            fork.destroy()

    def _result(
        self, gate: str, passed: bool, metrics: dict[str, float], detail: str, evidence: dict[str, Any]
    ) -> GateResult:
        path = self.cfg.evidence_dir / f"{gate}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"gate": gate, "passed": passed, "metrics": metrics, "detail": detail, **evidence}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))
        rel = path.resolve().relative_to(self.cfg.evidence_root.resolve()).as_posix()
        return GateResult(passed=passed, metrics=metrics, detail=detail, evidence_path=rel)

    def _solve(self, sb: Sandbox, module: str, probes: list[Probe], runs: int = 1) -> list[dict[str, Any]]:
        calls = [
            {"module": module, "fn": "solve", "args": [p.problem["prompt"]]} for p in probes for _ in range(runs)
        ]
        return run_calls(sb, calls, call_timeout_s=self.cfg.call_timeout_s)

    def _score(
        self, sb: Sandbox, items: list[tuple[dict[str, Any], str]], env: dict[str, str] | None = None
    ) -> list[Score]:
        return score(sb, GENERATOR, items, call_timeout_s=self.cfg.call_timeout_s, env=env)

    # -- gates -----------------------------------------------------------

    def g1(self) -> GateResult:
        """Static checks, sandbox creation, and gen determinism across hash seeds."""
        violations = [f"generator: {v}" for v in static_check(self.generator_src, kind="generator")]
        if self.solver_src is not None:
            violations += [f"solver: {v}" for v in static_check(self.solver_src, kind="solver")]
        if violations:
            detail = "static check failed:\n" + "\n".join(f"- {v}" for v in violations)
            return self._result("G1", False, {"violations": len(violations)}, detail, {"violations": violations})

        self.levels = _difficulty_levels(self.generator_src)
        pairs = probe_pairs(self.levels)
        files = {GENERATOR: self.generator_src.encode()}
        if self.solver_src is not None:
            files[SOLVER] = self.solver_src.encode()
        self.close()
        try:
            self._base = prepare_base(self.cfg.backend, self.cfg.spec, files)
            fork = self._base.fork()
            try:
                runs = [
                    generate(fork, GENERATOR, pairs, call_timeout_s=self.cfg.call_timeout_s, env={"PYTHONHASHSEED": h})
                    for h in G1_HASH_SEEDS
                ]
            finally:
                fork.destroy()
        except SandboxError as e:
            detail = f"sandbox: {e}"
            return self._result("G1", False, {"probes": len(pairs)}, detail, {"error": detail})

        failures: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        nondeterministic = 0
        for i, (s, d) in enumerate(pairs):
            gens = [run[i] for run in runs]
            hashes = [sha256_hex(canonical_json(g.problem)) if g.problem is not None else None for g in gens]
            records.append({"seed": s, "difficulty": d, "hashes": hashes, "problem": gens[0].problem})
            first = Probe(s, d, gens[0].problem) if gens[0].problem is not None else None
            bad = next((g for g in gens if g.error is not None), None)
            if bad is not None:
                failures.append(_failure(first, seed=s, difficulty=d, error=_error_text(bad.error, bad.message)))
            elif len(set(hashes)) != 1:
                nondeterministic += 1
                failures.append(
                    _failure(first, got=[g.problem["answer"] for g in gens], error="nondeterministic", hashes=hashes)
                )
        passed = not failures
        if passed:
            self.probes = [Probe(s, d, runs[0][i].problem) for i, (s, d) in enumerate(pairs)]  # type: ignore[arg-type]
        metrics = {"probes": len(pairs), "nondeterministic": nondeterministic}
        return self._result("G1", passed, metrics, _summarise(failures), {"failures": failures, "probes": records})

    def g2(self) -> GateResult:
        """Reference solve scores 1.0 on every probe; independent solver agrees on >= 95%."""
        if self.solver_src is None:
            raise ValueError("G2 needs an independent solver")
        failures: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        with self._fork() as sb:
            for p in self.probes:  # one probe per batch so the first failure stops the gate
                outs = self._solve(sb, GENERATOR, [p], runs=G2_REFERENCE_RUNS)
                solved = [o for o in outs if o.get("ok")]
                scores = self._score(sb, [(p.problem, _answer_tag(o["value"])) for o in solved])
                record = {"seed": p.seed, "difficulty": p.difficulty, "outputs": outs}
                records.append(record | {"scores": [asdict(sc) for sc in scores]})
                for o in outs:
                    if not o.get("ok"):
                        failures.append(_failure(p, error=_call_error("solve", o)))
                for o, sc in zip(solved, scores, strict=True):
                    if sc.error is not None or sc.value != 1.0:
                        err = _error_text(sc.error, sc.message)
                        failures.append(_failure(p, got=o["value"], check=sc.value, error=err and f"check {err}"))
                if failures:
                    break
            if failures:
                detail = _summarise(failures, "reference solve does not pass")
                metrics = {"reference_failures": len(failures)}
                return self._result("G2", False, metrics, detail, {"failures": failures, "reference": records})

            outs = self._solve(sb, SOLVER, self.probes)
            items = [
                (p.problem, _answer_tag(o["value"]) if o.get("ok") else "")
                for p, o in zip(self.probes, outs, strict=True)
            ]
            scores = self._score(sb, items)
        disagreements = []
        for p, o, sc in zip(self.probes, outs, scores, strict=True):
            if sc.value != 1.0:
                err = _call_error("solve", o) if not o.get("ok") else _error_text(sc.error, sc.message)
                disagreements.append(_failure(p, got=o.get("value"), check=sc.value, error=err))
        agreement = (len(self.probes) - len(disagreements)) / len(self.probes)
        passed = agreement >= G2_MIN_AGREEMENT
        header = "" if passed else f"independent solver agrees on {agreement:.4f} < {G2_MIN_AGREEMENT} of probes"
        evidence = {"failures": [], "disagreements": disagreements, "reference": records, "independent": outs}
        metrics = {"reference_failures": 0, "independent_agreement": agreement}
        return self._result("G2", passed, metrics, _summarise(disagreements, header), evidence)

    def g3(self) -> GateResult:
        """``check`` returns identical values across hash seeds for fixed responses."""
        mid = math.ceil(self.levels / 2)
        probes = [p for p in self.probes if p.difficulty == mid]
        answers = {p.seed: p.problem["answer"] for p in probes}
        with self._fork() as sb:
            refs = self._solve(sb, GENERATOR, probes)
            groups: list[tuple[Probe, str]] = []
            for p, r in zip(probes, refs, strict=True):
                groups.append((p, _answer_tag(r["value"]) if r.get("ok") else _answer_tag("")))
                groups.append((p, "<answer></answer>"))
                other = answers[(p.seed + 1) % len(PROBE_SEEDS)]
                if other != p.problem["answer"]:
                    groups.append((p, _answer_tag(other)))
            items = [(p.problem, resp) for p, resp in groups]
            runs = [self._score(sb, items, env={"PYTHONHASHSEED": h}) for h in G3_HASH_SEEDS]
        failures = [
            _failure(p, error=_call_error("solve", r))
            for p, r in zip(probes, refs, strict=True)
            if not r.get("ok")
        ]
        records = []
        nondeterministic = errors = 0
        for i, (p, resp) in enumerate(groups):
            values = [run[i].value for run in runs]
            errs = [_error_text(run[i].error, run[i].message) for run in runs]
            records.append({"seed": p.seed, "difficulty": p.difficulty, "response": resp, "values": values})
            if any(errs):
                errors += 1
                failures.append(_failure(p, got=resp, check=values[0], error=next(e for e in errs if e)))
            elif len(set(values)) != 1:
                nondeterministic += 1
                failures.append(_failure(p, got=resp, check=values[0], error="nondeterministic", values=values))
        metrics = {"groups": len(groups), "nondeterministic": nondeterministic, "errors": errors}
        evidence = {"failures": failures, "groups": records}
        return self._result("G3", not failures, metrics, _summarise(failures), evidence)

    def g4(self) -> GateResult:
        """Null and trivial responses score exactly 0.0 on every probe."""
        items: list[tuple[Probe, str]] = []
        for p in self.probes:
            nulls = ["", "<answer></answer>", "<answer> </answer>", "<answer>0</answer>", "<answer>None</answer>"]
            nulls += ["<answer>I don't know</answer>", _answer_tag(p.problem["prompt"])]
            expected = p.problem["answer"].strip()
            items += [(p, r) for r in nulls if extract_answer(r) != expected]
        with self._fork() as sb:
            scores = self._score(sb, [(p.problem, r) for p, r in items])
        failures = [
            _failure(p, got=r, check=sc.value) for (p, r), sc in zip(items, scores, strict=True) if sc.value != 0.0
        ]
        metrics = {"false_accepts": len(failures), "responses": len(items)}
        return self._result("G4", not failures, metrics, _summarise(failures), {"failures": failures})

    def g5(self) -> GateResult:
        """Single-point mutants of ``solve`` must fail at least one probe."""
        found = mutants(self.generator_src)
        if self._base is None or not self.probes:
            raise RuntimeError("G1 must pass before later gates run")
        for i, m in enumerate(found):
            self._base.write_file(f"mutants/m{i}.py", m.source.encode())
        records: list[dict[str, Any]] = []
        with self._fork() as sb:
            for i, m in enumerate(found):
                kill = self._kill(sb, f"mutants/m{i}.py")
                records.append({"index": i, "op": m.op, "lineno": m.lineno, "killed": kill is not None, "by": kill})
        total = len(found)
        killed = sum(r["killed"] for r in records)
        rate = killed / total if total else 0.0
        self.mutant_kill_rate = rate
        survivors = [(r["op"], r["lineno"]) for r in records if not r["killed"]]
        lines = []
        if total < G5_MIN_MUTANTS:
            lines.append(f"insufficient mutation sites: {total}")
        elif rate < G5_MIN_KILL_RATE:
            lines.append(f"kill rate {rate:.4f} < {G5_MIN_KILL_RATE}")
        if survivors:
            lines.append("surviving mutants (op, lineno): " + ", ".join(f"({op}, {ln})" for op, ln in survivors))
        passed = total >= G5_MIN_MUTANTS and rate >= G5_MIN_KILL_RATE
        metrics = {"mutants": total, "killed": killed, "kill_rate": rate}
        return self._result("G5", passed, metrics, "\n".join(lines), {"mutants": records})

    def _kill(self, sb: Sandbox, module: str) -> dict[str, Any] | None:
        """Return the first probe failure of a mutant, or None if it survives every probe."""
        # A first single-probe batch kills most mutants cheaply and bounds a hanging mutant to one timeout.
        for chunk in (self.probes[:1], self.probes[1:]):
            if not chunk:
                continue
            outs = self._solve(sb, module, chunk)
            solved = [(p, o) for p, o in zip(chunk, outs, strict=True) if o.get("ok")]
            for p, o in zip(chunk, outs, strict=True):
                if not o.get("ok"):
                    return _failure(p, error=_call_error("solve", o))
            scores = self._score(sb, [(p.problem, _answer_tag(o["value"])) for p, o in solved])
            for (p, o), sc in zip(solved, scores, strict=True):
                if sc.value < 1.0:
                    return _failure(p, got=o["value"], check=sc.value, error=_error_text(sc.error, sc.message))
        return None


def run_gates(
    generator_src: str, solver_src: str | None, *, cfg: GateConfig, stop_after: str | None = None
) -> GateRun:
    """Run G1..G5 in order, stopping at the first failure or after ``stop_after``."""
    if stop_after is not None and stop_after not in GATES:
        raise ValueError(f"unknown gate: {stop_after!r}")
    gates: dict[str, GateResult] = {}
    with Validator(cfg, generator_src, solver_src) as v:
        for gate in GATES:
            result = getattr(v, gate.lower())()
            gates[gate] = result
            if not result.passed or gate == stop_after:
                break
        return GateRun(gates=gates, probes=list(v.probes), levels=v.levels, mutant_kill_rate=v.mutant_kill_rate)


def validate_module(generator_src: str, solver_src: str, *, cfg: GateConfig) -> dict[str, GateResult]:
    """Run G1..G5 and stop at the first failure; later gates are absent from the result."""
    return run_gates(generator_src, solver_src, cfg=cfg).gates
