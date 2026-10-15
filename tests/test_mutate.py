from __future__ import annotations

import ast
from collections import Counter

from envforge.procedural.mutate import Mutant, mutants

SOURCE = '''\
import random

NAME = "toy"


def gen(seed, difficulty):
    rng = random.Random(seed)
    a = rng.randint(1, 9) + 2
    return {"prompt": str(a), "answer": str(a), "params": {}}


def solve(prompt):
    a, b, x, y = map(int, prompt.split())
    c = x < y
    if c:
        return str(min(a + b, 3))
    return str(a)


def check(problem, response):
    return 1.0 if response.strip() == problem["answer"] else 0.0
'''

HELPER_SOURCE = '''\
def gen(seed, difficulty):
    return {"prompt": "1", "answer": "1", "params": {}}


def helper(n):
    return n * 2


def unused(n):
    return n - 7


def solve(prompt):
    return str(helper(int(prompt)))


def check(problem, response):
    return float(response == problem["answer"])
'''


def _function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    (fn,) = [stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef) and stmt.name == name]
    return ast.unparse(fn)


def test_exact_op_multiset() -> None:
    ops = Counter(m.op for m in mutants(SOURCE))
    assert ops == Counter(["Lt->LtE", "negate_test", "min->max", "Add->Sub", "const+1", "const-1"])


def test_mutants_compile_and_differ() -> None:
    original = ast.unparse(ast.parse(SOURCE))
    for m in mutants(SOURCE):
        compile(m.source, "<mutant>", "exec")
        assert m.source != original
        assert 12 <= m.lineno <= 17


def test_gen_and_check_untouched() -> None:
    for m in mutants(SOURCE):
        for name in ("gen", "check"):
            assert _function_text(m.source, name) == _function_text(SOURCE, name)


def test_const_mutants_shift_value() -> None:
    by_op = {m.op: m for m in mutants(SOURCE)}
    assert "min(a + b, 4)" in _function_text(by_op["const+1"].source, "solve")
    assert "min(a + b, 2)" in _function_text(by_op["const-1"].source, "solve")
    assert "max(a + b, 3)" in _function_text(by_op["min->max"].source, "solve")
    assert "if not c:" in _function_text(by_op["negate_test"].source, "solve")


def test_transitive_helper_is_mutated() -> None:
    result = mutants(HELPER_SOURCE)
    assert Counter(m.op for m in result) == Counter(["Mult->Add", "const+1", "const-1"])
    for m in result:
        assert _function_text(m.source, "unused") == _function_text(HELPER_SOURCE, "unused")
        assert _function_text(m.source, "solve") == _function_text(HELPER_SOURCE, "solve")


def test_bool_constants_are_not_int_constants() -> None:
    source = "def solve(prompt):\n    flag = True\n    return str(flag)\n"
    assert mutants(source) == []


def test_boolop_and_not() -> None:
    source = "def solve(prompt):\n    a = not prompt\n    return str(a and prompt or a)\n"
    ops = Counter(m.op for m in mutants(source))
    assert ops == Counter(["drop_not", "Or->And", "And->Or"])


def test_no_solve_gives_no_mutants() -> None:
    assert mutants("def gen(seed, difficulty):\n    return 1 + 2\n") == []


def test_limit_sampling_is_deterministic() -> None:
    full = mutants(SOURCE)
    assert len(full) == 6
    first = mutants(SOURCE, limit=3, seed=7)
    second = mutants(SOURCE, limit=3, seed=7)
    assert first == second
    assert len(first) == 3
    assert all(isinstance(m, Mutant) for m in first)
    assert set(first) <= set(full)
    assert first == sorted(first, key=lambda m: (m.lineno, m.op))
