"""Packaging of a validated, profiled generator into store records."""

from __future__ import annotations

import sys

from envforge.answer import task_messages
from envforge.hashing import sha256_hex, short_id
from envforge.procedural.dedup import DedupRecord, DedupResult, recheck
from envforge.procedural.gates import Probe
from envforge.procedural.profile import LevelProfile
from envforge.procedural.seeds import Seed
from envforge.prompts import prompts_hash
from envforge.sandbox import SandboxBackend, SandboxSpec
from envforge.schemas import (
    Budgets,
    Environment,
    GateResult,
    Message,
    Provenance,
    ReferenceSolution,
    StateRef,
    Task,
    ValidationReport,
    Verifier,
)
from envforge.store import Store


def env_id_for(generator_src: str) -> str:
    """Content-hash id; rerunning a seed that yields the same module rewrites the same env."""
    return short_id("env", "procedural", generator_src)


def dedup_candidates(store: Store, seed_id: str, generator_src: str) -> list[DedupRecord]:
    """Stored envs a new module is compared against.

    A rerun of the same seed that yields the same module is idempotent, so that
    env is left out; the identical module from a different seed still counts.
    """
    own = env_id_for(generator_src)
    return [
        DedupRecord(*r)
        for r in store.accepted_dedup()
        if not (r[0] == own and store.load_env(own).provenance.seed_id == seed_id)
    ]


def split_for(env_id: str) -> str:
    """Generator-level split so train/eval/holdout never share a generator."""
    bucket = int(sha256_hex(env_id), 16) % 10
    return "holdout" if bucket == 0 else "eval" if bucket == 1 else "train"


def package(
    *,
    seed: Seed,
    store: Store,
    backend: SandboxBackend,
    spec: SandboxSpec,
    generator_src: str,
    solver_src: str,
    generator_model: str,
    pipeline_version: str,
    created_at: str,
    gates: dict[str, GateResult],
    mutant_kill_rate: float,
    probes: list[Probe],
    levels: int,
    dedup: DedupResult,
    profiled: list[LevelProfile],
    total_cost: float,
) -> list[str] | None:
    """Write env, verifier, tasks and reports under the store lock.

    Returns the task ids, or None when the dedup re-check finds that a
    near-duplicate was accepted concurrently.
    """
    env_id = env_id_for(generator_src)
    with store.lock:
        accepted = dedup_candidates(store, seed.id, generator_src)
        if not recheck(dedup.embedding, dedup.minhash, accepted).passed:
            return None

        tier = backend.isolation_tier
        if tier == "docker":
            image = spec.image
        else:
            image = f"process:python{sys.version_info.major}.{sys.version_info.minor}"
        src_hash = sha256_hex(generator_src)
        env = Environment(
            env_id=env_id,
            family="procedural",
            image=image,
            isolation_tier=tier,  # type: ignore[arg-type]
            tools=[],
            state_init=StateRef(kind="generator", ref="generator.py", params={"difficulty_levels": levels}),
            network_policy="none",
            harness_profiles=["single_turn_answer_tag"],
            provenance=Provenance(
                seed_id=seed.id,
                seed_source=seed.source,
                license=seed.license,
                generator_model=generator_model,
                prompts_hash=prompts_hash(),
                pipeline_version=pipeline_version,
                created_at=created_at,
            ),
        )
        verifier = Verifier(
            verifier_id=short_id("ver", env_id, "check"),
            kind="value_match",
            code_ref=f"generator.py::check@sha256:{src_hash}",
            hidden_checks=[],
            rubric=None,
            deterministic=True,
            partial_credit=False,
        )
        store.save_env(env, generator_src, solver_src, embedding=dedup.embedding, minhash=dedup.minhash)
        store.save_verifier(verifier)

        split = split_for(env_id)
        cost_each = total_cost / len(profiled)
        task_ids: list[str] = []
        for lp in profiled:
            d = lp.difficulty
            exemplar = next(p for p in probes if p.seed == 0 and p.difficulty == d).problem["prompt"]
            task = Task(
                task_id=short_id("task", env_id, str(d)),
                env_id=env_id,
                prompt=[Message(**m) for m in task_messages(exemplar)],
                difficulty_params={"difficulty": d},
                reference=ReferenceSolution(
                    kind="program", code_ref=f"generator.py::solve@sha256:{src_hash}", entrypoint="solve"
                ),
                verifier_id=verifier.verifier_id,
                budgets=Budgets(),
                tags=[*seed.tags, seed.id],
                split=split,  # type: ignore[arg-type]
            )
            report = ValidationReport(
                task_id=task.task_id,
                verifier_id=verifier.verifier_id,
                pipeline_version=pipeline_version,
                gates={**gates, "G10": dedup.gate, "G9": lp.gate},
                mutant_kill_rate=mutant_kill_rate,
                solver_panel={**lp.panel, lp.policy_key: lp.policy},
                band=lp.band,  # type: ignore[arg-type]
                near_dup_max_sim=dedup.near_dup_max_sim,
                contamination=dedup.contamination,
                cost_usd=cost_each,
            )
            store.save_task(task, regression=lp.band != "target")
            store.save_report(report)
            task_ids.append(task.task_id)
        return task_ids
