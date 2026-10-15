"""Sandbox protocol shared by the process and Docker backends.

Generated code never runs in the host process. It runs through ``runner.py``
inside a sandbox created by a backend returned from :func:`make_backend`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from envforge.config import SandboxConfig


@dataclass(frozen=True)
class SandboxSpec:
    """Resources and image for one sandbox. ``image`` is used by the Docker backend only."""

    image: str
    cpu: float
    memory_mb: int
    pids: int


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one command run inside a sandbox."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    duration_s: float


class SandboxError(RuntimeError):
    """A sandbox operation failed."""


class SandboxUnavailable(SandboxError):
    """The requested sandbox backend cannot run on this host."""


class Sandbox(Protocol):
    """An isolated working directory that runs commands and holds files."""

    workdir: str

    def exec(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` in ``workdir``. ``argv[0] == "python"`` selects the backend's interpreter."""
        ...

    def write_file(self, rel: str, data: bytes) -> None:
        """Write ``data`` to ``rel`` under ``workdir``, creating parent directories."""
        ...

    def read_file(self, rel: str) -> bytes:
        """Return the content of ``rel`` under ``workdir``."""
        ...

    def snapshot(self) -> str:
        """Capture the current state and return the snapshot id."""
        ...

    def fork(self) -> Sandbox:
        """Return a new sandbox that starts from a copy of this one."""
        ...

    def destroy(self) -> None:
        """Release every resource held by this sandbox."""
        ...


class SandboxBackend(Protocol):
    """Factory for sandboxes of one isolation tier."""

    isolation_tier: str

    def create(self, spec: SandboxSpec) -> Sandbox:
        """Create a new empty sandbox."""
        ...


def make_backend(cfg: SandboxConfig) -> SandboxBackend:
    """Return the backend named by ``cfg.backend``."""
    if cfg.backend == "process":
        from envforge.sandbox.process import ProcessBackend

        return ProcessBackend()
    if cfg.backend == "docker":
        from envforge.sandbox.docker import DockerBackend

        return DockerBackend()
    raise SandboxError(f"unknown sandbox backend: {cfg.backend!r}")


__all__ = [
    "ExecResult",
    "Sandbox",
    "SandboxBackend",
    "SandboxError",
    "SandboxSpec",
    "SandboxUnavailable",
    "make_backend",
]
