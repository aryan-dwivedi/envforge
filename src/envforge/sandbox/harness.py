"""Host-side driver for ``runner.py`` inside a sandbox."""

from __future__ import annotations

import importlib.resources
import json
from typing import Any

from envforge.sandbox import Sandbox, SandboxBackend, SandboxSpec

MAX_BATCH = 200
OUTER_TIMEOUT_SLACK_S = 10


def _parse(stdout: bytes, n: int) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(stdout)
    except (ValueError, UnicodeDecodeError):
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != n or not all(isinstance(r, dict) for r in results):
        return None
    return results


def _run_batch(
    sb: Sandbox, batch: list[dict[str, Any]], call_timeout_s: float, env: dict[str, str] | None
) -> list[dict[str, Any]]:
    request = json.dumps({"timeout_s": call_timeout_s, "calls": batch}).encode()
    res = sb.exec(
        ["python", "runner.py"],
        stdin=request,
        timeout_s=call_timeout_s * len(batch) + OUTER_TIMEOUT_SLACK_S,
        env=env,
    )
    if res.timed_out:
        return [{"ok": False, "error": "timeout"} for _ in batch]
    results = _parse(res.stdout, len(batch))
    if results is None:
        return [{"ok": False, "error": "runner_crash"} for _ in batch]
    return results


def run_calls(
    sb: Sandbox,
    calls: list[dict[str, Any]],
    *,
    call_timeout_s: float,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run ``calls`` through the runner in batches of at most ``MAX_BATCH``; one result per call.

    An outer timeout or unparseable runner output turns every result of that
    batch into ``{"ok": False, "error": "timeout"}`` or ``{"ok": False, "error":
    "runner_crash"}`` instead of raising.
    """
    results: list[dict[str, Any]] = []
    for i in range(0, len(calls), MAX_BATCH):
        results.extend(_run_batch(sb, calls[i : i + MAX_BATCH], call_timeout_s, env))
    return results


def runner_source() -> bytes:
    """Return the bytes of the packaged ``runner.py``."""
    return (importlib.resources.files("envforge.sandbox") / "runner.py").read_bytes()


def prepare_base(backend: SandboxBackend, spec: SandboxSpec, files: dict[str, bytes]) -> Sandbox:
    """Create a sandbox holding ``runner.py`` and ``files``."""
    sb = backend.create(spec)
    try:
        sb.write_file("runner.py", runner_source())
        for rel, data in files.items():
            sb.write_file(rel, data)
    except BaseException:
        sb.destroy()
        raise
    return sb
