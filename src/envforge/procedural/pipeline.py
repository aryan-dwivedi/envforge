"""Synthesis pipeline: build, repair and validate one seed into stored tasks.

Flow per seed: the builder writes a generator module, G1 runs, an independent
solver is written from G1 examples, G1-G5 run, and failures go to the
repairer. A passing module then goes through G10 (dedup and contamination)
before G9 (difficulty profiling), and is packaged under the store lock.
"""

from __future__ import annotations

import ast
import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from envforge.config import Config, ConfigError
from envforge.llm import LLM, ChatModel, ChatResult, Embedder, EmbedModel
from envforge.procedural.contract import extract_python
from envforge.procedural.dedup import ContaminationIndex, g10
from envforge.procedural.gates import GateConfig, GateRun, Probe, run_gates
from envforge.procedural.package import dedup_candidates, package
from envforge.procedural.profile import profile
from envforge.procedural.seeds import Seed
from envforge.prompts import render_builder, render_independent_solver, render_repair
from envforge.sandbox import SandboxSpec, make_backend
from envforge.sandbox.harness import prepare_base
from envforge.schemas import Budgets, GateResult, RejectedRun
from envforge.store import Store

PIPELINE_VERSION = "procedural@0.1.0"


@dataclass
class Models:
    builder: ChatModel
    repairer: ChatModel
    independent_solver: ChatModel
    policy: ChatModel
    panel: list[tuple[str, ChatModel]]  # (family, model)
    embedder: EmbedModel
    contamination: ContaminationIndex


def build_models(cfg: Config) -> Models:
    """Build every role client from config; fail fast on bad suites or a dead embedder."""
    roles = cfg.roles
    repairer = roles.repairer if roles.repairer is not None else roles.builder
    contamination = ContaminationIndex.from_suites(cfg.contamination.suites)
    embedder = Embedder(roles.embedder)
    try:
        embedder.embed(["envforge probe"])
    except Exception as e:
        raise ConfigError(f"embedder endpoint failed: {e}") from e
    return Models(
        builder=LLM(roles.builder),
        repairer=LLM(repairer),
        independent_solver=LLM(roles.independent_solver),
        policy=LLM(roles.policy),
        panel=[(p.family, LLM(p)) for p in roles.panel],
        embedder=embedder,
        contamination=contamination,
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _Cost:
    total: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def chat(self, model: ChatModel, prompt: str) -> str:
        result: ChatResult = model.chat([{"role": "user", "content": prompt}])
        self.add(result.cost_usd)
        return result.text

    def add(self, usd: float) -> None:
        with self.lock:
            self.total += usd


@dataclass
class _Validated:
    src: str
    solver_src: str
    run: GateRun
    generator_model: str


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _description(source: str) -> str:
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DESCRIPTION" for t in stmt.targets
        ):
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                return stmt.value.value
    raise ValueError("DESCRIPTION not found")


def _probe(probes: list[Probe], seed: int, difficulty: int) -> Probe:
    return next(p for p in probes if p.seed == seed and p.difficulty == difficulty)


def _first_failure(gates: dict[str, GateResult]) -> tuple[str, GateResult]:
    return next((g, r) for g, r in gates.items() if not r.passed)


def _gate_config(cfg: Config, store: Store, run_key: str) -> GateConfig:
    sb = cfg.sandbox
    spec = SandboxSpec(image=sb.image, cpu=sb.cpu, memory_mb=sb.memory_mb, pids=sb.pids)
    return GateConfig(make_backend(sb), spec, cfg.limits.call_timeout_s, store.evidence_dir(run_key), store.root)


def _validate(
    seed: Seed, cfg: Config, store: Store, models: Models, cost: _Cost, attempts: list[dict[str, GateResult]]
) -> _Validated:
    src = extract_python(cost.chat(models.builder, render_builder(seed)))
    author = models.builder.model
    n_attempts = cfg.run.max_repairs + 1
    for n in range(1, n_attempts + 1):
        gate_cfg = _gate_config(cfg, store, f"{seed.id}/attempt{n}")
        if src is None:
            src = ""
            gates = {"G1": GateResult(passed=False, detail="no python block")}
            solver_src = ""
            run = GateRun(gates=gates)
        else:
            run = run_gates(src, None, cfg=gate_cfg, stop_after="G1")
            solver_src = ""
            if run.gates["G1"].passed:
                levels = run.levels
                examples = [
                    (p.problem["prompt"], p.problem["answer"])
                    for p in (_probe(run.probes, 0, d) for d in (1, math.ceil(levels / 2), levels))
                ]
                reply = cost.chat(models.independent_solver, render_independent_solver(_description(src), examples))
                solver_src = extract_python(reply) or ""
                run = run_gates(src, solver_src, cfg=gate_cfg)
            gates = run.gates
        attempts.append(dict(gates))
        if all(r.passed for r in gates.values()) and "G5" in gates:
            return _Validated(src, solver_src, run, author)
        gate, result = _first_failure(gates)
        if n == n_attempts:
            raise _Rejected(f"gate {gate} failed after {n} attempts")
        src = extract_python(cost.chat(models.repairer, render_repair(src, gate, result.detail)))
        author = models.repairer.model
    raise AssertionError("unreachable")


def _run(
    seed: Seed, cfg: Config, store: Store, models: Models, cost: _Cost, attempts: list[dict[str, GateResult]]
) -> list[str]:
    v = _validate(seed, cfg, store, models, cost, attempts)
    probes, levels = v.run.probes, v.run.levels
    mid = math.ceil(levels / 2)
    dedup = g10(
        description=_description(v.src),
        dedup_prompts=[_probe(probes, s, mid).problem["prompt"] for s in (0, 1, 2)],
        probe_prompts=[p.problem["prompt"] for p in probes],
        embedder=models.embedder,
        accepted=dedup_candidates(store, seed.id, v.src),
        index=models.contamination,
    )
    attempts[-1]["G10"] = dedup.gate
    if not dedup.gate.passed:
        raise _Rejected(f"G10: {dedup.gate.detail}")

    gate_cfg = _gate_config(cfg, store, f"{seed.id}/attempt{len(attempts)}")
    base = prepare_base(gate_cfg.backend, gate_cfg.spec, {"generator.py": v.src.encode()})
    try:
        prof = profile(
            base,
            levels=levels,
            panel=models.panel,
            policy=models.policy,
            call_timeout_s=cfg.limits.call_timeout_s,
            max_tokens=Budgets().max_tokens,
            concurrency=cfg.run.llm_concurrency,
        )
    finally:
        base.destroy()
    cost.add(prof.cost_usd)
    bands = ", ".join(f"{lp.difficulty}:{lp.band}" for lp in prof.levels)
    has_target = any(lp.band == "target" for lp in prof.levels)
    attempts[-1]["G9"] = GateResult(passed=has_target, detail=f"bands {bands}")
    kept = [lp for lp in prof.levels if lp.band in ("target", "too_easy")]
    if not kept:
        raise _Rejected("G9: no target band level")
    task_ids = package(
        seed=seed,
        store=store,
        backend=gate_cfg.backend,
        spec=gate_cfg.spec,
        generator_src=v.src,
        solver_src=v.solver_src,
        generator_model=v.generator_model,
        pipeline_version=PIPELINE_VERSION,
        created_at=now_iso(),
        gates=v.run.gates,
        mutant_kill_rate=v.run.mutant_kill_rate or 0.0,
        probes=probes,
        levels=levels,
        dedup=dedup,
        profiled=kept,
        total_cost=cost.total,
    )
    if task_ids is None:
        raise _Rejected("G10: dedup_race")
    if not has_target:
        raise _Rejected("G9: no target band level")
    return task_ids


def run_seed(seed: Seed, cfg: Config, store: Store, models: Models) -> list[str] | RejectedRun:
    """Return the accepted task ids (target and too_easy) or the saved rejection."""
    cost = _Cost()
    attempts: list[dict[str, GateResult]] = []
    try:
        task_ids = _run(seed, cfg, store, models, cost, attempts)
        store.clear_rejected(seed.id)
        return task_ids
    except _Rejected as e:
        reason = e.reason
    except Exception as e:
        reason = f"error: {type(e).__name__}: {e}"
    rejected = RejectedRun(
        seed_id=seed.id, reason=reason, attempts=attempts, cost_usd=cost.total, created_at=now_iso()
    )
    store.save_rejected(rejected)
    return rejected
