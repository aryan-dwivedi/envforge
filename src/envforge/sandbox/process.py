"""Local process sandbox backend."""

from __future__ import annotations

import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from envforge.sandbox import ExecResult, SandboxError, SandboxSpec

_BASE_ENV = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
_DRAIN_TIMEOUT_S = 5.0


def _mkdtemp() -> str:
    return tempfile.mkdtemp(prefix="envforge-")


def _limits(timeout_s: float, memory_mb: int) -> Callable[[], None]:
    cpu_s = math.ceil(timeout_s) + 1

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        if sys.platform == "linux":
            mem = memory_mb * 2**20
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    return apply


class ProcessSandbox:
    """Sandbox over a temporary directory, running commands as host subprocesses.

    Development only. There is no network isolation and no filesystem isolation
    beyond the working directory; commands run as the current user with a
    minimal environment, a CPU time limit, and (on Linux) an address space
    limit. Use the Docker backend for anything that claims isolation.
    """

    isolation_tier = "process"

    def __init__(self, workdir: str, spec: SandboxSpec) -> None:
        self.workdir = workdir
        self._root = Path(workdir).resolve()
        self._spec = spec
        self._snapshots: list[str] = []

    def _path(self, rel: str) -> Path:
        path = (self._root / rel).resolve()
        if path == self._root or not path.is_relative_to(self._root):
            raise SandboxError(f"path escapes sandbox workdir: {rel!r}")
        return path

    def exec(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if not argv:
            raise SandboxError("empty argv")
        if argv[0] == "python":
            # `-s -P` rather than `-I`: `-I` implies `-E`, which would drop the
            # PYTHONHASHSEED the determinism gates set. The environment is
            # already exactly _BASE_ENV plus the caller's vars, so `-E` adds nothing.
            argv = [sys.executable, "-s", "-P", *argv[1:]]
        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=self.workdir,
            env=_BASE_ENV | (env or {}),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_limits(timeout_s, self._spec.memory_mb),
        )
        try:
            out, err = proc.communicate(input=stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = self._drain(proc)
            return ExecResult(-9, out, err, True, time.monotonic() - start)
        return ExecResult(proc.returncode, out, err, False, time.monotonic() - start)

    @staticmethod
    def _drain(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
        try:
            return proc.communicate(timeout=_DRAIN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # A process that left the group can hold the pipes open; stop waiting on it.
            proc.kill()
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            proc.wait()
            return b"", b""

    def write_file(self, rel: str, data: bytes) -> None:
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_file(self, rel: str) -> bytes:
        path = self._path(rel)
        try:
            return path.read_bytes()
        except OSError as e:
            raise SandboxError(f"cannot read {rel!r}: {e.strerror or e}") from e

    def _copy(self) -> str:
        dst = _mkdtemp()
        shutil.copytree(self.workdir, dst, symlinks=True, dirs_exist_ok=True)
        return dst

    def snapshot(self) -> str:
        snap = self._copy()
        self._snapshots.append(snap)
        return snap

    def fork(self) -> ProcessSandbox:
        return ProcessSandbox(self._copy(), self._spec)

    def destroy(self) -> None:
        for path in [self.workdir, *self._snapshots]:
            shutil.rmtree(path, ignore_errors=True)
        self._snapshots.clear()


class ProcessBackend:
    """Backend creating :class:`ProcessSandbox` instances. Development only, no network isolation."""

    isolation_tier = "process"

    def create(self, spec: SandboxSpec) -> ProcessSandbox:
        return ProcessSandbox(_mkdtemp(), spec)
