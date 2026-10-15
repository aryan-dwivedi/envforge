"""Docker sandbox backend driven through the ``docker`` CLI."""

from __future__ import annotations

import dataclasses
import posixpath
import shutil
import subprocess
import time
import uuid

from envforge.sandbox import ExecResult, SandboxError, SandboxSpec, SandboxUnavailable

_WORKDIR = "/work"
_ADMIN_TIMEOUT_S = 300.0


def _docker(args: list[str], *, stdin: bytes = b"") -> bytes:
    """Run one administrative docker command and return its stdout."""
    try:
        proc = subprocess.run(
            ["docker", *args],
            input=stdin,
            capture_output=True,
            timeout=_ADMIN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"docker {args[0]} timed out") from e
    except OSError as e:
        raise SandboxUnavailable(f"docker CLI not runnable: {e}") from e
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise SandboxError(f"docker {args[0]} failed ({proc.returncode}): {msg}")
    return proc.stdout


def _container_path(rel: str) -> str:
    norm = posixpath.normpath(posixpath.join(_WORKDIR, rel))
    if not norm.startswith(_WORKDIR + "/"):
        raise SandboxError(f"path escapes sandbox workdir: {rel!r}")
    return norm


class DockerSandbox:
    """Sandbox in a container with no network, no capabilities, and CPU, memory and pid limits."""

    isolation_tier = "docker"

    def __init__(self, container_id: str, spec: SandboxSpec, owned_tag: str | None = None) -> None:
        self.workdir = _WORKDIR
        self.container_id = container_id
        self._spec = spec
        self._owned_tags: list[str] = [owned_tag] if owned_tag else []
        self._dead: str | None = None

    def _check_alive(self) -> None:
        if self._dead is not None:
            raise SandboxError(self._dead)

    def exec(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self._check_alive()
        if not argv:
            raise SandboxError("empty argv")
        cmd = ["docker", "exec", "-i", "-w", _WORKDIR]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [self.container_id, *argv]
        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", self.container_id], capture_output=True, timeout=_ADMIN_TIMEOUT_S)
            self._dead = "sandbox killed after timeout"
            return ExecResult(-9, e.stdout or b"", e.stderr or b"", True, time.monotonic() - start)
        return ExecResult(proc.returncode, proc.stdout, proc.stderr, False, time.monotonic() - start)

    def write_file(self, rel: str, data: bytes) -> None:
        self._check_alive()
        path = _container_path(rel)
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        _docker(["exec", "-i", self.container_id, "sh", "-c", script, "sh", path], stdin=data)

    def read_file(self, rel: str) -> bytes:
        self._check_alive()
        return _docker(["exec", self.container_id, "cat", _container_path(rel)])

    def snapshot(self) -> str:
        self._check_alive()
        tag = f"envforge-snap:{uuid.uuid4().hex}"
        _docker(["commit", self.container_id, tag])
        return tag

    def fork(self) -> DockerSandbox:
        tag = self.snapshot()
        try:
            return DockerBackend()._start(dataclasses.replace(self._spec, image=tag), owned_tag=tag)
        except BaseException:
            subprocess.run(["docker", "rmi", tag], capture_output=True, timeout=_ADMIN_TIMEOUT_S)
            raise

    def destroy(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True, timeout=_ADMIN_TIMEOUT_S)
        for tag in self._owned_tags:
            subprocess.run(["docker", "rmi", tag], capture_output=True, timeout=_ADMIN_TIMEOUT_S)
        self._owned_tags.clear()
        self._dead = "sandbox destroyed"


class DockerBackend:
    """Backend creating :class:`DockerSandbox` containers through the ``docker`` CLI."""

    isolation_tier = "docker"

    def create(self, spec: SandboxSpec) -> DockerSandbox:
        return self._start(spec, owned_tag=None)

    def _start(self, spec: SandboxSpec, *, owned_tag: str | None) -> DockerSandbox:
        if shutil.which("docker") is None:
            raise SandboxUnavailable("docker CLI not found")
        args = [
            "run",
            "-d",
            "--network",
            "none",
            "--cpus",
            str(spec.cpu),
            "--memory",
            f"{spec.memory_mb}m",
            "--pids-limit",
            str(spec.pids),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-w",
            _WORKDIR,
            spec.image,
            "sleep",
            "infinity",
        ]
        container_id = _docker(args).decode().strip()
        sb = DockerSandbox(container_id, spec, owned_tag)
        try:
            _docker(["exec", container_id, "mkdir", "-p", _WORKDIR])
        except BaseException:
            sb.destroy()
            raise
        return sb
