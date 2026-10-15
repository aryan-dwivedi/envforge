"""Configuration models and loader.

API key environment variables are resolved lazily by the model clients, not
here, so `envforge validate` works without any keys set.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(ValueError):
    """Invalid or unusable configuration."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleConfig(_Model):
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 8000
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0


class PanelRoleConfig(RoleConfig):
    family: str


class EmbedderConfig(_Model):
    base_url: str
    model: str
    api_key_env: str


class SandboxConfig(_Model):
    backend: Literal["process", "docker"]
    image: str = "python:3.13-slim"
    cpu: float
    memory_mb: int
    pids: int


class LimitsConfig(_Model):
    call_timeout_s: float


class RunConfig(_Model):
    llm_concurrency: int
    max_repairs: int


class SuiteConfig(_Model):
    name: str
    path: Path
    field: str


class ContaminationConfig(_Model):
    suites: list[SuiteConfig]


class RolesConfig(_Model):
    builder: RoleConfig
    repairer: RoleConfig | None = None
    independent_solver: RoleConfig
    panel: list[PanelRoleConfig]
    policy: RoleConfig
    embedder: EmbedderConfig


class Config(_Model):
    store_dir: Path
    sandbox: SandboxConfig
    limits: LimitsConfig
    run: RunConfig
    roles: RolesConfig
    contamination: ContaminationConfig


def _resolve(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: Path) -> Config:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {path}: {e}") from e

    roles = cfg.roles
    if roles.repairer is None:
        roles.repairer = roles.builder.model_copy()
    if roles.independent_solver.model in {roles.builder.model, roles.repairer.model}:
        raise ConfigError("independent_solver must use a different model than builder/repairer")
    if len(roles.panel) < 2 or len({p.family for p in roles.panel}) < 2:
        raise ConfigError("panel needs >=2 entries from >=2 model families")

    base = path.resolve().parent
    cfg.store_dir = _resolve(base, cfg.store_dir)
    for suite in cfg.contamination.suites:
        suite.path = _resolve(base, suite.path)
    return cfg
