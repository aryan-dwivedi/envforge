"""Single-point AST mutants of a solver, used to measure how many wrong solvers a checker rejects.

Target functions are `solve` plus every module-level function transitively called by bare name
from `solve`; `gen` and `check` are never targets. Each site yields one mutant per applicable
operator:

| Site                         | Op names                                                   |
|------------------------------|------------------------------------------------------------|
| `BinOp`, `AugAssign`         | `Add->Sub`, `Sub->Add`, `Mult->Add`, `Div->Mult`,          |
|                              | `FloorDiv->Mult`, `Mod->FloorDiv`, `Pow->Mult`             |
| `Compare` (each operator)    | `Lt->LtE`, `LtE->Lt`, `Gt->GtE`, `GtE->Gt`, `Eq->NotEq`,   |
|                              | `NotEq->Eq`                                                |
| int `Constant` (not bool)    | `const+1`, `const-1`                                       |
| `BoolOp`                     | `And->Or`, `Or->And`                                       |
| `UnaryOp(Not)`               | `drop_not` (replace `not x` with `x`)                      |
| `If`, `While`, `IfExp` test  | `negate_test` (replace `t` with `not (t)`)                 |
| `Call` to `min` / `max`      | `min->max`, `max->min`                                     |
"""

from __future__ import annotations

import ast
import random
from collections.abc import Callable
from dataclasses import dataclass

_BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
    ast.Pow: ast.Mult,
}

_CMPOP_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_MINMAX_SWAPS = {"min": "max", "max": "min"}
_EXCLUDED = frozenset({"gen", "check"})

# An edit receives the located site node and returns its replacement (or the node itself after an in-place change).
_Edit = Callable[[ast.AST], ast.AST]


@dataclass(frozen=True)
class Mutant:
    op: str
    lineno: int
    source: str


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {stmt.name: stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef)}


def _target_names(tree: ast.Module) -> list[str]:
    """Return `solve` and its transitive module-level callees, in module order."""
    funcs = _module_functions(tree)
    if "solve" not in funcs:
        return []
    seen = {"solve"}
    stack = ["solve"]
    while stack:
        for node in ast.walk(funcs[stack.pop()]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in funcs and name not in seen and name not in _EXCLUDED:
                    seen.add(name)
                    stack.append(name)
    return sorted(seen, key=lambda name: funcs[name].lineno)


def _set_binop(new: type[ast.operator]) -> _Edit:
    def edit(node: ast.AST) -> ast.AST:
        assert isinstance(node, ast.BinOp | ast.AugAssign)
        node.op = new()
        return node

    return edit


def _set_cmpop(index: int, new: type[ast.cmpop]) -> _Edit:
    def edit(node: ast.AST) -> ast.AST:
        assert isinstance(node, ast.Compare)
        node.ops[index] = new()
        return node

    return edit


def _shift_const(delta: int) -> _Edit:
    def edit(node: ast.AST) -> ast.AST:
        assert isinstance(node, ast.Constant)
        node.value = node.value + delta
        return node

    return edit


def _swap_boolop(node: ast.AST) -> ast.AST:
    assert isinstance(node, ast.BoolOp)
    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    return node


def _drop_not(node: ast.AST) -> ast.AST:
    assert isinstance(node, ast.UnaryOp)
    return node.operand


def _negate_test(node: ast.AST) -> ast.AST:
    assert isinstance(node, ast.If | ast.While | ast.IfExp)
    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    return node


def _swap_minmax(node: ast.AST) -> ast.AST:
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    node.func.id = _MINMAX_SWAPS[node.func.id]
    return node


def _site_ops(node: ast.AST) -> list[tuple[str, _Edit]]:
    """Return (op name, edit) pairs applicable to one node."""
    ops: list[tuple[str, _Edit]] = []
    if isinstance(node, ast.BinOp | ast.AugAssign):
        new = _BINOP_SWAPS.get(type(node.op))
        if new is not None:
            ops.append((f"{type(node.op).__name__}->{new.__name__}", _set_binop(new)))
    elif isinstance(node, ast.Compare):
        for index, cmpop in enumerate(node.ops):
            new_cmp = _CMPOP_SWAPS.get(type(cmpop))
            if new_cmp is not None:
                ops.append((f"{type(cmpop).__name__}->{new_cmp.__name__}", _set_cmpop(index, new_cmp)))
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            ops.append(("const+1", _shift_const(1)))
            ops.append(("const-1", _shift_const(-1)))
    elif isinstance(node, ast.BoolOp):
        ops.append(("And->Or" if isinstance(node.op, ast.And) else "Or->And", _swap_boolop))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            ops.append(("drop_not", _drop_not))
    elif isinstance(node, ast.If | ast.While | ast.IfExp):
        ops.append(("negate_test", _negate_test))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _MINMAX_SWAPS:
            ops.append((f"{node.func.id}->{_MINMAX_SWAPS[node.func.id]}", _swap_minmax))
    return ops


class _Replace(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.AST) -> None:
        self.target = target
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)


def _apply(source: str, func_name: str, walk_index: int, edit: _Edit) -> tuple[int, str] | None:
    """Re-parse `source`, edit the site at `walk_index` inside `func_name`, and return (lineno, new source)."""
    tree = ast.parse(source)
    site = list(ast.walk(_module_functions(tree)[func_name]))[walk_index]
    lineno = getattr(site, "lineno", 0)
    replacement = edit(site)
    if replacement is not site:
        tree = _Replace(site, replacement).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        new_source = ast.unparse(tree)
        compile(new_source, "<mutant>", "exec")
    except (SyntaxError, ValueError):
        return None
    return lineno, new_source


def mutants(source: str, *, limit: int = 50, seed: int = 0) -> list[Mutant]:
    """Return single-point mutants of the solver functions in `source`."""
    tree = ast.parse(source)
    original = ast.unparse(tree)
    funcs = _module_functions(tree)
    result: list[Mutant] = []
    for func_name in _target_names(tree):
        for walk_index, node in enumerate(ast.walk(funcs[func_name])):
            for op, edit in _site_ops(node):
                applied = _apply(source, func_name, walk_index, edit)
                if applied is None or applied[1] == original:
                    continue
                result.append(Mutant(op=op, lineno=applied[0], source=applied[1]))
    if len(result) > limit:
        result = sorted(random.Random(seed).sample(result, limit), key=lambda m: (m.lineno, m.op))
    return result
