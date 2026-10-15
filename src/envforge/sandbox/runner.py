"""Call runner executed inside a sandbox as ``python runner.py``.

This file is copied verbatim into sandboxes and exported packages, so it
imports only the standard library.

Protocol. stdin holds one JSON object::

    {"timeout_s": float, "calls": [{"module": "<rel path>", "fn": "gen"|"solve"|"check", "args": [...]}]}

stdout receives exactly one JSON line ``{"results": [...]}`` with one entry per
call, either ``{"ok": true, "value": ...}`` or ``{"ok": false, "error":
"timeout"|"exception"|"bad_return"|"import_error", "message": "..."}``.

Every call and every module import runs under a real-time interval timer.
The timer raises ``_Timeout``, a ``BaseException`` subclass, so a broad
``except Exception`` in the called code cannot swallow it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import sys
from typing import Any

_MESSAGE_LIMIT = 500
_MIN_TIMEOUT_S = 0.001
_FNS = ("gen", "solve", "check")


class _Timeout(BaseException):
    """Raised by the interval timer when a call exceeds its budget."""


def _on_alarm(signum: int, frame: object) -> None:
    raise _Timeout()


def _fail(error: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "message": message[:_MESSAGE_LIMIT]}


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _timed(fn: Any, args: list[Any], timeout_s: float) -> Any:
    """Call ``fn(*args)`` with stdout/stderr captured and the interval timer armed."""
    sink_out, sink_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _check_return(fn: str, value: Any) -> str | None:
    """Return a description of what is wrong with ``value``, or None when it is valid."""
    if fn == "gen":
        if not isinstance(value, dict):
            return f"gen must return a dict, got {type(value).__name__}"
        for key, kind in (("prompt", str), ("answer", str), ("params", dict)):
            if key not in value:
                return f"gen result is missing {key!r}"
            if not isinstance(value[key], kind):
                return f"gen result {key!r} must be {kind.__name__}, got {type(value[key]).__name__}"
        if not _is_json(value):
            return "gen result is not JSON-serializable"
        return None
    if fn == "solve":
        if not isinstance(value, str):
            return f"solve must return str, got {type(value).__name__}"
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"check must return int or float, got {type(value).__name__}"
    if not 0 <= value <= 1:
        return f"check must return a value in [0, 1], got {value!r}"
    return None


class _Modules:
    """Imports each module once, cached by path."""

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s
        self._cache: dict[str, tuple[Any, str | None]] = {}

    def get(self, path: str) -> tuple[Any, str | None]:
        if path not in self._cache:
            self._cache[path] = self._load(path, len(self._cache))
        return self._cache[path]

    def _load(self, path: str, idx: int) -> tuple[Any, str | None]:
        name = f"m{idx}"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                return None, f"cannot load module from {path!r}"
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            _timed(spec.loader.exec_module, [module], self._timeout_s)
        except _Timeout:
            sys.modules.pop(name, None)
            return None, "timeout during import"
        except BaseException as exc:
            sys.modules.pop(name, None)
            return None, _describe(exc)
        return module, None


def _run_call(call: Any, modules: _Modules, timeout_s: float) -> dict[str, Any]:
    if not isinstance(call, dict):
        return _fail("exception", "call must be an object")
    path, fn_name, args = call.get("module"), call.get("fn"), call.get("args", [])
    if not isinstance(path, str) or fn_name not in _FNS or not isinstance(args, list):
        return _fail("exception", f"malformed call: module={path!r} fn={fn_name!r}")
    module, import_error = modules.get(path)
    if import_error is not None:
        return _fail("import_error", import_error)
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        return _fail("exception", f"module {path!r} has no callable {fn_name!r}")
    try:
        value = _timed(fn, args, timeout_s)
    except _Timeout:
        return _fail("timeout", f"{fn_name} exceeded {timeout_s}s")
    except BaseException as exc:
        return _fail("exception", _describe(exc))
    problem = _check_return(fn_name, value)
    if problem is not None:
        return _fail("bad_return", problem)
    if fn_name == "check":
        value = float(value)
    return {"ok": True, "value": value}


def main() -> None:
    sys.dont_write_bytecode = True
    # Keep the protocol channel private: stray writes to fd 1 from called code land on stderr.
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    request = json.loads(sys.stdin.read())
    timeout_s = max(float(request["timeout_s"]), _MIN_TIMEOUT_S)
    calls = request["calls"]
    signal.signal(signal.SIGALRM, _on_alarm)
    modules = _Modules(timeout_s)
    results = []
    for call in calls:
        try:
            results.append(_run_call(call, modules, timeout_s))
        except _Timeout:
            # The timer fired while the call was being torn down.
            results.append(_fail("timeout", "timer fired after the call returned"))
    out.write(json.dumps({"results": results}) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
