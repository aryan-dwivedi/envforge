"""Answer-format prompt and extraction. Stdlib-only; copied into OpenEnv exports."""

from __future__ import annotations

import re

FORMAT_INSTRUCTION = (
    "Solve the problem. Give only your final answer inside <answer></answer> tags at the end of your response."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """Return the stripped content of the last <answer> block, or None if absent."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def task_messages(prompt: str) -> list[dict]:
    """Build the system+user chat messages for a single-turn task."""
    return [{"role": "system", "content": FORMAT_INSTRUCTION}, {"role": "user", "content": prompt}]
