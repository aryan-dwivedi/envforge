"""Command-line interface for envforge."""

from __future__ import annotations

import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer

from envforge.audit import audit_ingest, audit_sample
from envforge.config import Config, ConfigError, load_config
from envforge.export.openenv import export_openenv
from envforge.export.trl import export_trl
from envforge.llm import LLM
from envforge.procedural.gates import GATES, RATE_METRICS, GateConfig, validate_module
from envforge.procedural.pipeline import build_models, run_seed
from envforge.procedural.seeds import load_seeds
from envforge.sandbox import SandboxSpec, make_backend
from envforge.sandbox.process import ProcessBackend
from envforge.schemas import GateResult, RejectedRun
from envforge.stats import compute_stats, format_stats
from envforge.store import Store

app = typer.Typer(no_args_is_help=True, help="Validation-first compiler for RL environments and evals.")
export_app = typer.Typer(no_args_is_help=True, help="Export accepted tasks.")
audit_app = typer.Typer(no_args_is_help=True, help="Human audit of verifier decisions.")
app.add_typer(export_app, name="export")
app.add_typer(audit_app, name="audit")

ConfigOpt = Annotated[Path, typer.Option("--config", help="envforge.toml.")]

DEFAULT_SPEC = SandboxSpec(image="python:3.13-slim", cpu=1.0, memory_mb=1024, pids=256)
DEFAULT_CALL_TIMEOUT_S = 5.0


@app.callback()
def main() -> None:
    """Validation-first compiler for RL environments and evals."""


def _format_metric(key: str, value: float) -> str:
    if key not in RATE_METRICS and float(value).is_integer():
        return f"{key}={int(value)}"
    return f"{key}={value:.4f}"


def format_gate(gate: str, result: GateResult) -> str:
    """Render one gate as ``<gate> PASS|FAIL k=v ...``, followed by the detail on failure."""
    status = "PASS" if result.passed else "FAIL"
    line = " ".join([gate, status, *(_format_metric(k, v) for k, v in result.metrics.items())])
    if not result.passed and result.detail:
        line += "\n" + result.detail
    return line


def _gate_config(config: Path | None) -> GateConfig:
    if config is None:
        root = Path(tempfile.mkdtemp(prefix="envforge-evidence-"))
        typer.echo(f"evidence: {root}")
        return GateConfig(ProcessBackend(), DEFAULT_SPEC, DEFAULT_CALL_TIMEOUT_S, root, root)
    cfg = load_config(config)
    sb = cfg.sandbox
    spec = SandboxSpec(image=sb.image, cpu=sb.cpu, memory_mb=sb.memory_mb, pids=sb.pids)
    evidence_dir = cfg.store_dir / "evidence" / "validate"
    typer.echo(f"evidence: {evidence_dir}")
    return GateConfig(make_backend(sb), spec, cfg.limits.call_timeout_s, evidence_dir, cfg.store_dir)


@app.command()
def validate(
    generator: Annotated[Path, typer.Argument(help="Generator module (.py).", exists=True, dir_okay=False)],
    solver: Annotated[Path, typer.Option("--solver", help="Independent solver module (.py).", exists=True)],
    config: Annotated[Path | None, typer.Option("--config", help="envforge.toml; defaults to a local run.")] = None,
) -> None:
    """Run gates G1-G5 on a generator module without any LLM; exit 0 iff all pass."""
    try:
        gate_cfg = _gate_config(config)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    results = validate_module(generator.read_text(), solver.read_text(), cfg=gate_cfg)
    for gate, result in results.items():
        typer.echo(format_gate(gate, result))
    ok = all(g in results and results[g].passed for g in GATES)
    raise typer.Exit(0 if ok else 1)


def _format_outcome(seed_id: str, outcome: list[str] | RejectedRun, store: Store) -> str:
    if isinstance(outcome, RejectedRun):
        return f"{seed_id} REJECTED {outcome.reason}"
    bands = Counter(store.load_report(t).band for t in outcome)
    order = sorted(bands, key=lambda b: (b != "target", b))
    return f"{seed_id} ACCEPTED tasks={len(outcome)} bands=" + ",".join(f"{b}:{bands[b]}" for b in order)


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", help="envforge.toml.")],
    seeds: Annotated[str | None, typer.Option("--seeds", help="Comma-separated seed ids.")] = None,
    all_seeds: Annotated[bool, typer.Option("--all", help="Run every built-in seed.")] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, help="Seeds run in parallel.")] = 4,
) -> None:
    """Synthesize, validate and store environments for the selected seeds."""
    if not seeds and not all_seeds:
        typer.echo("pass --seeds or --all", err=True)
        raise typer.Exit(2)
    catalogue = {s.id: s for s in load_seeds()}
    if all_seeds:
        selected = list(catalogue.values())
    else:
        ids = [s.strip() for s in (seeds or "").split(",") if s.strip()]
        unknown = [i for i in ids if i not in catalogue]
        if unknown:
            typer.echo(f"unknown seed: {', '.join(unknown)}", err=True)
            raise typer.Exit(2)
        selected = [catalogue[i] for i in ids]
    try:
        cfg = load_config(config)
        models = build_models(cfg)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    store = Store(cfg.store_dir)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_seed, s, cfg, store, models): s for s in selected}
        for fut in as_completed(futures):
            typer.echo(_format_outcome(futures[fut].id, fut.result(), store))


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e


@app.command()
def stats(config: ConfigOpt) -> None:
    """Print run statistics for the configured store."""
    cfg = _load(config)
    for line in format_stats(compute_stats(Store(cfg.store_dir))):
        typer.echo(line)


@export_app.command("trl")
def export_trl_cmd(
    config: ConfigOpt,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    split: Annotated[str, typer.Option("--split")] = "train",
    instances_per_task: Annotated[int, typer.Option("--instances-per-task", min=1)] = 64,
    include_regression: Annotated[bool, typer.Option("--include-regression")] = False,
) -> None:
    """Write a TRL dataset (data.jsonl + manifest.json); score with export.trl.make_reward_func."""
    cfg = _load(config)
    result = export_trl(
        Store(cfg.store_dir),
        out,
        split=split,
        instances_per_task=instances_per_task,
        include_regression=include_regression,
        call_timeout_s=cfg.limits.call_timeout_s,
    )
    typer.echo(f"rows={result['rows']} tasks={len(result['tasks'])} gen_failures={result['gen_failures']} out={out}")


@export_app.command("openenv")
def export_openenv_cmd(
    config: ConfigOpt,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    split: Annotated[str, typer.Option("--split")] = "train",
    include_regression: Annotated[bool, typer.Option("--include-regression")] = False,
) -> None:
    """Write a self-contained OpenEnv server package."""
    cfg = _load(config)
    try:
        result = export_openenv(Store(cfg.store_dir), out, split=split, include_regression=include_regression)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"tasks={result['tasks']} out={result['out']}")


@audit_app.command("sample")
def audit_sample_cmd(
    config: ConfigOpt,
    out: Annotated[Path, typer.Option("--out", help="Audit JSONL to write.")],
    n: Annotated[int, typer.Option("--n", min=1)] = 200,
) -> None:
    """Sample policy responses with their rewards for hand labelling."""
    cfg = _load(config)
    try:
        rows = audit_sample(
            Store(cfg.store_dir),
            LLM(cfg.roles.policy),
            n=n,
            out=out,
            call_timeout_s=cfg.limits.call_timeout_s,
            concurrency=cfg.run.llm_concurrency,
        )
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"rows={rows} out={out}")


@audit_app.command("ingest")
def audit_ingest_cmd(config: ConfigOpt, path: Annotated[Path, typer.Argument(help="Labelled audit JSONL.")]) -> None:
    """Record hand labels on the reports; exit 1 if any false accept was found."""
    cfg = _load(config)
    totals = audit_ingest(Store(cfg.store_dir), path)
    typer.echo(" ".join(f"{k}={v}" for k, v in totals.items()))
    raise typer.Exit(1 if totals["false_accepts"] > 0 else 0)
