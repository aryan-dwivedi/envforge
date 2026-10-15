from __future__ import annotations

from envforge.procedural.contract import BANNED_NAMES, IMPORT_ALLOWLIST, extract_python, static_check
from envforge.procedural.seeds import load_seeds

GOOD = '''\
import random
from collections import Counter

NAME = "modular-sum"
DESCRIPTION = "Sum a list of integers modulo m."
DIFFICULTY_LEVELS = 5


def gen(seed, difficulty):
    rng = random.Random(seed)
    xs = [rng.randint(0, 99) for _ in range(difficulty + 2)]
    m = rng.randint(2, 50)
    return {"prompt": f"{xs} mod {m}", "answer": str(sum(xs) % m), "params": {"m": m}}


def solve(prompt):
    left, m = prompt.rsplit(" mod ", 1)
    xs = [int(t) for t in left.strip("[]").split(",")]
    return str(sum(xs) % int(m))


def check(problem, response):
    return 1.0 if response.strip() == problem["answer"] else 0.0
'''


def test_good_generator_passes() -> None:
    assert static_check(GOOD, kind="generator") == []


def test_constants_match_contract() -> None:
    assert "math" in IMPORT_ALLOWLIST and "os" not in IMPORT_ALLOWLIST
    assert {"open", "eval", "getattr", "__import__"} <= BANNED_NAMES


def test_import_os_flagged() -> None:
    violations = static_check("import os\n" + GOOD, kind="generator")
    assert any("import not allowed: os" in v for v in violations)


def test_from_subprocess_flagged() -> None:
    violations = static_check("from subprocess import run\n" + GOOD, kind="generator")
    assert any("import not allowed: subprocess" in v for v in violations)


def test_dotted_import_uses_root() -> None:
    violations = static_check("import os.path\ndef solve(prompt):\n    return prompt\n", kind="solver")
    assert any("import not allowed: os" in v for v in violations)


def test_relative_import_flagged() -> None:
    violations = static_check("from . import x\ndef solve(prompt):\n    return prompt\n", kind="solver")
    assert any("relative import" in v for v in violations)


def test_banned_names_flagged() -> None:
    source = "def solve(prompt):\n    open('x')\n    return eval(prompt)\n"
    violations = static_check(source, kind="solver")
    assert any("banned name: open" in v for v in violations)
    assert any("banned name: eval" in v for v in violations)


def test_dunder_attribute_flagged() -> None:
    source = "def solve(prompt):\n    return str(prompt.__class__)\n"
    violations = static_check(source, kind="solver")
    assert any("__class__" in v for v in violations)


def test_missing_name_flagged() -> None:
    source = GOOD.replace('NAME = "modular-sum"\n', "")
    violations = static_check(source, kind="generator")
    assert any("NAME" in v and "DESCRIPTION" not in v for v in violations)


def test_difficulty_levels_out_of_range() -> None:
    source = GOOD.replace("DIFFICULTY_LEVELS = 5", "DIFFICULTY_LEVELS = 11")
    assert static_check(source, kind="generator") == ["DIFFICULTY_LEVELS must be in 1..10, got 11"]
    source = GOOD.replace("DIFFICULTY_LEVELS = 5", "DIFFICULTY_LEVELS = True")
    assert static_check(source, kind="generator") == ["DIFFICULTY_LEVELS must be an int constant"]


def test_missing_functions_for_generator() -> None:
    source = GOOD.split("def gen")[0] + "def solve(prompt):\n    return prompt\n"
    violations = static_check(source, kind="generator")
    assert "missing function: gen(seed, difficulty)" in violations
    assert "missing function: check(problem, response)" in violations
    assert not any("solve" in v for v in violations)


def test_wrong_parameters_flagged() -> None:
    source = GOOD.replace("def check(problem, response):", "def check(problem):")
    violations = static_check(source, kind="generator")
    assert any("function check must have parameters (problem, response)" in v for v in violations)


def test_solver_kind_only_needs_solve() -> None:
    assert static_check("import math\ndef solve(prompt):\n    return prompt\n", kind="solver") == []
    assert static_check("def gen(seed, difficulty):\n    return {}\n", kind="solver") == [
        "missing function: solve(prompt)"
    ]


def test_syntax_error_reported() -> None:
    violations = static_check("def solve(prompt)\n    return prompt\n", kind="solver")
    assert len(violations) == 1
    assert violations[0].startswith("syntax error")


def test_extract_python_picks_last_block() -> None:
    text = (
        "First:\n```python\nx = 1\n```\n"
        "Other:\n```text\nnot code\n```\n"
        "Final:\n```py\ny = 2\n```\n"
        "Trailing:\n```\nz = 3\n```\n"
    )
    assert extract_python(text) == "y = 2\n"


def test_extract_python_absent() -> None:
    assert extract_python("no code here") is None
    assert extract_python("```js\nlet x = 1;\n```\n") is None


def test_load_seeds() -> None:
    seeds = load_seeds()
    assert len(seeds) == 60
    assert len({s.id for s in seeds}) == 60
    assert seeds[0].id == "modular-sum"
    assert seeds[-1].id == "shoelace-area"
    for s in seeds:
        assert s.license == "CC0-1.0"
        assert s.source == "envforge-builtin"
        assert 1 <= len(s.tags) <= 3
        assert s.title and s.description
