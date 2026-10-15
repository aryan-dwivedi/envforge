"""Markdown prompt templates for the builder, repairer and independent solver."""

from __future__ import annotations

from importlib.resources import files

from envforge.hashing import sha256_hex
from envforge.procedural.contract import BANNED_NAMES, CONTRACT_TEXT, IMPORT_ALLOWLIST
from envforge.procedural.seeds import Seed

TEMPLATE_NAMES = ("builder.md", "repair.md", "independent_solver.md")
DETAIL_MAX_CHARS = 4000


def _template(name: str) -> str:
    return (files("envforge.prompts") / name).read_text(encoding="utf-8")


def _allowlist() -> str:
    return ", ".join(sorted(IMPORT_ALLOWLIST))


def prompts_hash() -> str:
    """sha256 over the concatenated template texts, in TEMPLATE_NAMES order."""
    return sha256_hex("".join(_template(n) for n in TEMPLATE_NAMES))


def render_builder(seed: Seed) -> str:
    return _template("builder.md").format(
        title=seed.title,
        description=seed.description,
        contract=CONTRACT_TEXT,
        allowlist=_allowlist(),
        banned=", ".join(sorted(BANNED_NAMES)),
    )


def render_repair(module_source: str, gate: str, detail: str) -> str:
    return _template("repair.md").format(
        module_source=module_source,
        gate=gate,
        detail=detail[:DETAIL_MAX_CHARS],
    )


def render_independent_solver(description: str, examples: list[tuple[str, str]]) -> str:
    """`examples` are exactly 3 (prompt, answer) pairs; the builder's code is never shown."""
    if len(examples) != 3:
        raise ValueError(f"expected 3 examples, got {len(examples)}")
    blocks = [
        f"### Example {i}\n\nPrompt:\n\n```text\n{p}\n```\n\nAnswer:\n\n```text\n{a}\n```"
        for i, (p, a) in enumerate(examples, 1)
    ]
    return _template("independent_solver.md").format(
        description=description,
        examples="\n\n".join(blocks),
        allowlist=_allowlist(),
    )
