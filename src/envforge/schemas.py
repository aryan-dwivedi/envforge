"""Pydantic models for environments, tasks, verifiers and validation reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSpec(_Frozen):
    name: str
    description: str
    parameters: dict
    side: Literal["agent", "user"]


class StateRef(_Frozen):
    kind: Literal["db_snapshot", "repo_commit", "workbook", "generator"]
    ref: str
    params: dict = Field(default_factory=dict)


class Provenance(_Frozen):
    seed_id: str
    seed_source: str
    license: str
    generator_model: str
    prompts_hash: str
    pipeline_version: str
    created_at: str


class Message(_Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ReferenceSolution(_Frozen):
    kind: Literal["program", "trajectory"]
    code_ref: str
    entrypoint: str


class Budgets(_Frozen):
    max_turns: int = 1
    max_tokens: int = 4096
    wall_clock_s: float = 60.0
    submissions: int = 1


class Rubric(_Frozen):
    weights: dict[str, float]
    judge_model: str | None = None
    calibration_set_id: str | None = None


class Environment(_Frozen):
    env_id: str
    family: Literal["mcp_db_world", "repo_code", "procedural", "spreadsheet"]
    image: str
    isolation_tier: Literal["process", "docker", "gvisor", "microvm"]
    tools: list[ToolSpec]
    state_init: StateRef
    network_policy: str
    harness_profiles: list[str]
    provenance: Provenance


class Task(_Frozen):
    task_id: str
    env_id: str
    prompt: str | list[Message]
    difficulty_params: dict | None
    reference: ReferenceSolution
    verifier_id: str
    budgets: Budgets
    tags: list[str]
    split: Literal["train", "eval", "holdout"]


class Verifier(_Frozen):
    verifier_id: str
    kind: Literal["state_check", "tests_f2p_p2p", "value_match", "rubric", "hybrid"]
    code_ref: str
    hidden_checks: list[str]
    rubric: Rubric | None
    deterministic: bool
    partial_credit: bool


class GateResult(_Frozen):
    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    detail: str = ""
    evidence_path: str | None = None


class PassStats(_Frozen):
    n: int
    successes: int
    k: int
    pass_at_1: float
    pass_at_k: float


class AuditResult(_Frozen):
    n_audited: int
    false_accepts: int
    false_rejects: int
    auditor: str
    audited_at: str


class RunObservation(_Frozen):
    run_id: str
    observed_at: str
    n: int
    pass_rate: float
    flags: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Mutable so that lifecycle observations and human audits can be appended."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    verifier_id: str
    pipeline_version: str
    gates: dict[str, GateResult]
    mutant_kill_rate: float
    hack_probes: dict[str, bool] = Field(default_factory=dict)
    solver_panel: dict[str, PassStats]
    band: Literal["too_easy", "target", "too_hard", "unsolved"]
    near_dup_max_sim: float
    contamination: dict[str, float]
    cost_usd: float
    human_audit: AuditResult | None = None
    lifecycle: list[RunObservation] = Field(default_factory=list)


class RejectedRun(_Frozen):
    seed_id: str
    reason: str
    attempts: list[dict[str, GateResult]]
    cost_usd: float
    created_at: str
