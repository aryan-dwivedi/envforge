"""Module contract for procedural generators and solvers, with static source checks."""

from __future__ import annotations

import ast
import re
from typing import Literal

CONTRACT_TEXT = """\
NAME: str
DESCRIPTION: str             # problem family description; shown to the independent solver
DIFFICULTY_LEVELS: int       # 1..10; valid difficulties are 1..DIFFICULTY_LEVELS
def gen(seed: int, difficulty: int) -> dict: ...  # {"prompt": str, "answer": str, "params": dict}; must use random.Random(seed) only
def solve(prompt: str) -> str: ...                # parses the prompt text only; returns bare answer string
def check(problem: dict, response: str) -> float: # problem = gen() output; response = extracted answer; returns value in [0,1]
"""  # noqa: E501

IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "math",
        "random",
        "itertools",
        "functools",
        "collections",
        "heapq",
        "fractions",
        "decimal",
        "string",
        "re",
        "json",
        "dataclasses",
        "typing",
        "bisect",
        "statistics",
        "operator",
        "enum",
    }
)

BANNED_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
    }
)

_GENERATOR_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "gen": ("seed", "difficulty"),
    "solve": ("prompt",),
    "check": ("problem", "response"),
}
_SOLVER_FUNCTIONS: dict[str, tuple[str, ...]] = {"solve": ("prompt",)}

_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _check_names(tree: ast.Module) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in IMPORT_ALLOWLIST:
                    violations.append(f"import not allowed: {root} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"relative import not allowed (line {node.lineno})")
            elif node.module is not None:
                root = node.module.split(".")[0]
                if root not in IMPORT_ALLOWLIST:
                    violations.append(f"import not allowed: {root} (line {node.lineno})")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                violations.append(f"banned name: {node.id} (line {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                violations.append(f"dunder attribute access not allowed: {node.attr} (line {node.lineno})")
    return violations


def _top_level_constants(tree: ast.Module) -> dict[str, ast.expr | None]:
    constants: dict[str, ast.expr | None] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            constants[stmt.target.id] = stmt.value
    return constants


def _check_constants(tree: ast.Module) -> list[str]:
    violations: list[str] = []
    constants = _top_level_constants(tree)
    for name in ("NAME", "DESCRIPTION"):
        value = constants.get(name)
        if name not in constants:
            violations.append(f"missing top-level {name} (str constant)")
        elif not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            violations.append(f"{name} must be a str constant")
    value = constants.get("DIFFICULTY_LEVELS")
    if "DIFFICULTY_LEVELS" not in constants:
        violations.append("missing top-level DIFFICULTY_LEVELS (int constant 1..10)")
    elif not (
        isinstance(value, ast.Constant) and isinstance(value.value, int) and not isinstance(value.value, bool)
    ):
        violations.append("DIFFICULTY_LEVELS must be an int constant")
    elif not 1 <= value.value <= 10:
        violations.append(f"DIFFICULTY_LEVELS must be in 1..10, got {value.value}")
    return violations


def _check_functions(tree: ast.Module, required: dict[str, tuple[str, ...]]) -> list[str]:
    violations: list[str] = []
    defs = {stmt.name: stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef)}
    for name, params in required.items():
        signature = f"{name}({', '.join(params)})"
        fn = defs.get(name)
        if fn is None:
            violations.append(f"missing function: {signature}")
            continue
        got = tuple(arg.arg for arg in [*fn.args.posonlyargs, *fn.args.args])
        if got != params:
            violations.append(f"function {name} must have parameters ({', '.join(params)}), got ({', '.join(got)})")
    return violations


def static_check(source: str, *, kind: Literal["generator", "solver"]) -> list[str]:
    """Return contract violations for `source`; an empty list means the source is acceptable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    except ValueError as exc:
        return [f"syntax error: {exc}"]
    violations = _check_names(tree)
    if kind == "generator":
        violations.extend(_check_constants(tree))
        violations.extend(_check_functions(tree, _GENERATOR_FUNCTIONS))
    elif kind == "solver":
        violations.extend(_check_functions(tree, _SOLVER_FUNCTIONS))
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    return violations


def extract_python(text: str) -> str | None:
    """Return the body of the last fenced code block tagged `python` or `py`, or None."""
    body: str | None = None
    for match in _FENCE_RE.finditer(text):
        info = match.group(1).strip().split()
        if info and info[0].lower() in ("python", "py"):
            body = match.group(2)
    return body
