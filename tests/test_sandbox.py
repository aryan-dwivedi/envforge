from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from envforge.config import SandboxConfig
from envforge.sandbox import ExecResult, SandboxError, SandboxSpec, SandboxUnavailable, make_backend
from envforge.sandbox.docker import DockerBackend
from envforge.sandbox.harness import MAX_BATCH, prepare_base, run_calls
from envforge.sandbox.process import ProcessBackend

SPEC = SandboxSpec(image="python:3.13-slim", cpu=1.0, memory_mb=512, pids=64)

GOOD_MODULE = b'''
import random

def gen(seed, difficulty):
    rng = random.Random(seed)
    a, b = rng.randint(1, 100), rng.randint(1, 100)
    print("noise on stdout")
    return {"prompt": f"{a}+{b}", "answer": str(a + b), "params": {"a": a, "b": b, "d": difficulty}}

def solve(prompt):
    a, b = prompt.split("+")
    return str(int(a) + int(b))

def check(problem, response):
    return 1.0 if response.strip() == problem["answer"] else 0
'''


@pytest.fixture
def sb():
    box = ProcessBackend().create(SPEC)
    try:
        yield box
    finally:
        box.destroy()


@pytest.fixture
def base():
    box = prepare_base(ProcessBackend(), SPEC, {"mod.py": GOOD_MODULE})
    try:
        yield box
    finally:
        box.destroy()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_make_backend_selects_backend():
    cfg = {"cpu": 1.0, "memory_mb": 512, "pids": 64}
    process = make_backend(SandboxConfig(backend="process", **cfg))
    docker = make_backend(SandboxConfig(backend="docker", **cfg))
    assert isinstance(process, ProcessBackend) and process.isolation_tier == "process"
    assert isinstance(docker, DockerBackend) and docker.isolation_tier == "docker"


def test_docker_backend_without_cli_is_unavailable(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SandboxUnavailable, match="docker CLI not found"):
        DockerBackend().create(SPEC)


def test_process_exec_runs_python_with_minimal_env(sb):
    code = (
        "import json, os, sys; "
        "print(json.dumps([dict(os.environ), sys.flags.no_user_site, sys.flags.safe_path]))"
    )
    res = sb.exec(["python", "-c", code], timeout_s=10, env={"EXTRA": "1"})
    assert res.exit_code == 0 and not res.timed_out
    env, no_user_site, safe_path = json.loads(res.stdout)
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["EXTRA"] == "1"
    assert "HOME" not in env
    assert no_user_site == 1 and safe_path is True


def test_process_exec_honors_pythonhashseed(sb):
    # The determinism gates rely on the chosen hash seed reaching the child.
    def run(seed: str) -> bytes:
        res = sb.exec(["python", "-c", "print(hash('envforge'))"], timeout_s=10, env={"PYTHONHASHSEED": seed})
        return res.stdout.strip()

    assert run("0") == run("0")
    assert run("0") != run("1")


def test_process_exec_passes_stdin_and_cwd(sb):
    sb.write_file("data.txt", b"from-file")
    code = "import sys; print(open('data.txt').read() + sys.stdin.read())"
    res = sb.exec(["python", "-c", code], stdin=b"+stdin", timeout_s=10)
    assert res.stdout.strip() == b"from-file+stdin"


def test_process_exec_timeout_kills_process_group(sb):
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "while True: pass\n"
    )
    start = time.monotonic()
    res = sb.exec(["python", "-c", code], timeout_s=1.5)
    assert res.timed_out and res.exit_code == -9
    assert time.monotonic() - start < 10
    child = int(res.stdout.split()[0])
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)


def test_process_spin_times_out(sb):
    res = sb.exec(["python", "-c", "while True: pass"], timeout_s=1)
    assert res.timed_out and res.exit_code == -9


def test_process_fork_isolates_writes(sb):
    sb.write_file("a.txt", b"base")
    fork = sb.fork()
    try:
        assert fork.workdir != sb.workdir
        assert fork.read_file("a.txt") == b"base"
        fork.write_file("a.txt", b"fork")
        fork.write_file("b.txt", b"new")
        assert sb.read_file("a.txt") == b"base"
        with pytest.raises(SandboxError):
            sb.read_file("b.txt")
    finally:
        fork.destroy()
    assert not Path(fork.workdir).exists()
    assert sb.read_file("a.txt") == b"base"


def test_process_snapshot_copies_state(sb):
    sb.write_file("a.txt", b"one")
    snap = sb.snapshot()
    sb.write_file("a.txt", b"two")
    assert (Path(snap) / "a.txt").read_bytes() == b"one"
    sb.destroy()
    assert not Path(snap).exists()


def test_process_write_file_creates_parents(sb):
    sb.write_file("mutants/m0.py", b"x = 1\n")
    assert sb.read_file("mutants/m0.py") == b"x = 1\n"


@pytest.mark.parametrize("rel", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
def test_process_path_escape_raises(sb, rel):
    with pytest.raises(SandboxError, match="escapes"):
        sb.write_file(rel, b"x")
    with pytest.raises(SandboxError, match="escapes"):
        sb.read_file(rel)


def test_process_symlink_escape_raises(sb, tmp_path):
    os.symlink(tmp_path, Path(sb.workdir) / "link")
    with pytest.raises(SandboxError, match="escapes"):
        sb.write_file("link/x.txt", b"x")


def test_runner_is_stdlib_only():
    src = (Path(__file__).parents[1] / "src" / "envforge" / "sandbox" / "runner.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module.split(".")[0])
    assert "envforge" not in names
    assert names <= set(sys.stdlib_module_names)


def test_runner_happy_path(base):
    calls = [
        {"module": "mod.py", "fn": "gen", "args": [3, 1]},
        {"module": "mod.py", "fn": "solve", "args": ["2+5"]},
        {"module": "mod.py", "fn": "check", "args": [{"answer": "7"}, " 7 "]},
        {"module": "mod.py", "fn": "check", "args": [{"answer": "7"}, "8"]},
    ]
    gen, solve, ok, wrong = run_calls(base, calls, call_timeout_s=5)
    assert gen["ok"] and set(gen["value"]) == {"prompt", "answer", "params"}
    assert gen["value"]["answer"] == str(gen["value"]["params"]["a"] + gen["value"]["params"]["b"])
    assert solve == {"ok": True, "value": "7"}
    assert ok == {"ok": True, "value": 1.0}
    assert wrong == {"ok": True, "value": 0.0}
    again = run_calls(base, calls[:1], call_timeout_s=5)
    assert again[0] == gen


def test_runner_per_call_timeout(base):
    base.write_file("spin.py", b"def solve(prompt):\n    while True:\n        pass\n")
    swallow = (
        b"def solve(prompt):\n"
        b"    while True:\n"
        b"        try:\n"
        b"            pass\n"
        b"        except Exception:\n"
        b"            pass\n"
    )
    base.write_file("swallow.py", swallow)
    calls = [
        {"module": "mod.py", "fn": "solve", "args": ["1+1"]},
        {"module": "spin.py", "fn": "solve", "args": ["x"]},
        {"module": "swallow.py", "fn": "solve", "args": ["x"]},
        {"module": "mod.py", "fn": "solve", "args": ["2+2"]},
    ]
    start = time.monotonic()
    first, spin, swallowed, last = run_calls(base, calls, call_timeout_s=0.5)
    assert time.monotonic() - start < 8
    assert first == {"ok": True, "value": "2"}
    assert spin["ok"] is False and spin["error"] == "timeout"
    assert swallowed["ok"] is False and swallowed["error"] == "timeout"
    assert last == {"ok": True, "value": "4"}


def test_runner_bad_return(base):
    base.write_file(
        "bad.py",
        b"def check(p, r):\n    return True\n"
        b"def solve(p):\n    return 3\n"
        b"def gen(seed, d):\n    return {'prompt': 'p', 'answer': 'a', 'params': {'x': object()}}\n",
    )
    base.write_file("range.py", b"def check(p, r):\n    return 1.5\n")
    calls = [
        {"module": "bad.py", "fn": "check", "args": [{}, ""]},
        {"module": "bad.py", "fn": "solve", "args": ["p"]},
        {"module": "bad.py", "fn": "gen", "args": [0, 1]},
        {"module": "range.py", "fn": "check", "args": [{}, ""]},
    ]
    results = run_calls(base, calls, call_timeout_s=5)
    assert [r["error"] for r in results] == ["bad_return"] * 4
    assert "bool" in results[0]["message"]


def test_runner_exception_message_truncated(base):
    base.write_file("boom.py", b"def solve(p):\n    raise ValueError('x' * 2000)\n")
    (res,) = run_calls(base, [{"module": "boom.py", "fn": "solve", "args": ["p"]}], call_timeout_s=5)
    assert res["ok"] is False and res["error"] == "exception"
    assert res["message"].startswith("ValueError: xxx") and len(res["message"]) == 500


def test_runner_import_error(base):
    base.write_file("broken.py", b"def gen(seed, d)\n    return {}\n")
    calls = [
        {"module": "broken.py", "fn": "gen", "args": [0, 1]},
        {"module": "mod.py", "fn": "solve", "args": ["1+2"]},
        {"module": "broken.py", "fn": "solve", "args": ["x"]},
    ]
    a, b, c = run_calls(base, calls, call_timeout_s=5)
    assert a["error"] == "import_error" and "SyntaxError" in a["message"]
    assert b == {"ok": True, "value": "3"}
    assert c["error"] == "import_error"


def test_runner_import_timeout(base):
    base.write_file("slow.py", b"while True:\n    pass\n")
    (res,) = run_calls(base, [{"module": "slow.py", "fn": "gen", "args": [0, 1]}], call_timeout_s=0.5)
    assert res["error"] == "import_error" and "timeout" in res["message"]


def test_harness_splits_batches(base):
    calls = [{"module": "mod.py", "fn": "solve", "args": [f"{i}+1"]} for i in range(MAX_BATCH + 5)]
    results = run_calls(base, calls, call_timeout_s=5)
    assert [r["value"] for r in results] == [str(i + 1) for i in range(MAX_BATCH + 5)]


def test_harness_runner_crash_on_hard_exit(base):
    base.write_file("exit.py", b"import os\ndef solve(p):\n    os._exit(3)\n")
    calls = [{"module": "mod.py", "fn": "solve", "args": ["1+1"]}, {"module": "exit.py", "fn": "solve", "args": ["x"]}]
    assert run_calls(base, calls, call_timeout_s=5) == [{"ok": False, "error": "runner_crash"}] * 2


def test_harness_stray_fd_write_does_not_corrupt_protocol(base):
    base.write_file("fd.py", b"import os\ndef solve(p):\n    os.write(1, b'garbage\\n')\n    return 'ok'\n")
    assert run_calls(base, [{"module": "fd.py", "fn": "solve", "args": ["x"]}], call_timeout_s=5) == [
        {"ok": True, "value": "ok"}
    ]


class _FakeSandbox:
    workdir = "/fake"

    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.timeouts: list[float] = []

    def exec(self, argv, *, stdin=b"", timeout_s, env=None):
        assert argv == ["python", "runner.py"]
        self.timeouts.append(timeout_s)
        return self.result


def test_harness_outer_timeout_and_garbage():
    calls = [{"module": "m.py", "fn": "solve", "args": ["x"]}] * 3
    timed_out = _FakeSandbox(ExecResult(-9, b"", b"", True, 1.0))
    assert run_calls(timed_out, calls, call_timeout_s=2) == [{"ok": False, "error": "timeout"}] * 3
    assert timed_out.timeouts == [2 * 3 + 10]
    garbage = _FakeSandbox(ExecResult(0, b"not json", b"", False, 1.0))
    assert run_calls(garbage, calls, call_timeout_s=2) == [{"ok": False, "error": "runner_crash"}] * 3
    short = _FakeSandbox(ExecResult(0, b'{"results": [{"ok": true, "value": "x"}]}', b"", False, 1.0))
    assert run_calls(short, calls, call_timeout_s=2) == [{"ok": False, "error": "runner_crash"}] * 3


docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not found")


@pytest.fixture
def dsb():
    box = DockerBackend().create(SPEC)
    try:
        yield box
    finally:
        box.destroy()


@pytest.mark.docker
@docker
def test_docker_exec_fork_destroy(dsb):
    dsb.write_file("dir/a.txt", b"base")
    res = dsb.exec(["python", "-c", "print(open('dir/a.txt').read())"], timeout_s=30)
    assert res.exit_code == 0 and res.stdout.strip() == b"base"
    fork = dsb.fork()
    try:
        fork.write_file("dir/a.txt", b"fork")
        assert fork.read_file("dir/a.txt") == b"fork"
        assert dsb.read_file("dir/a.txt") == b"base"
    finally:
        fork.destroy()
    with pytest.raises(SandboxError):
        fork.read_file("dir/a.txt")


@pytest.mark.docker
@docker
def test_docker_network_none(dsb):
    code = "import socket; socket.create_connection(('1.1.1.1', 53), 2)"
    res = dsb.exec(["python", "-c", code], timeout_s=30)
    assert res.exit_code != 0


@pytest.mark.docker
@docker
def test_docker_timeout_kills_sandbox(dsb):
    res = dsb.exec(["python", "-c", "while True: pass"], timeout_s=2)
    assert res.timed_out
    with pytest.raises(SandboxError, match="sandbox killed after timeout"):
        dsb.exec(["python", "-c", "pass"], timeout_s=5)


@pytest.mark.docker
@docker
def test_docker_runner_happy_path():
    box = prepare_base(DockerBackend(), SPEC, {"mod.py": GOOD_MODULE})
    try:
        (res,) = run_calls(box, [{"module": "mod.py", "fn": "solve", "args": ["2+3"]}], call_timeout_s=5)
        assert res == {"ok": True, "value": "5"}
    finally:
        box.destroy()
