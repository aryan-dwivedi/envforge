#!/usr/bin/env python3
"""Ratchet the size of every .py source under src/, tests/ and scripts/.

- A file at or under LIMIT lines passes.
- A file over LIMIT must be listed in .file-size-baseline and may not grow
  past its recorded count. Shrinking is always fine; --update lowers the
  recorded counts and drops files that fell under the limit.

    python3 scripts/check-file-size.py            # check, exit 1 on a violation
    python3 scripts/check-file-size.py --update   # lower the baseline

--update never raises a count or adds a file: growth is fixed by splitting the
file. --update --accept-growth exists only to seed a baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / ".file-size-baseline"
ROOTS = ("src", "tests", "scripts")
LIMIT = 500
SKIP_DIRS = {"__pycache__", ".venv", "build", "dist"}
HEADER = (
    "# Source files over the size limit, with their line count. A listed file may\n"
    f"# shrink but never grow; an unlisted file must stay at or under {LIMIT} lines.\n"
    "# Maintained by scripts/check-file-size.py.\n"
)


def sources() -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_DIRS.intersection(path.relative_to(REPO).parts):
                continue
            with path.open("rb") as fh:
                counts[path.relative_to(REPO).as_posix()] = sum(1 for _ in fh)
    return counts


def read_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    out: dict[str, int] = {}
    for line in BASELINE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        count, _, rel = line.partition(" ")
        out[rel] = int(count)
    return out


def write_baseline(counts: dict[str, int]) -> None:
    rows = "".join(f"{n} {rel}\n" for rel, n in sorted(counts.items()))
    BASELINE.write_text(HEADER + rows)


def check(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    errors = []
    for rel, n in sorted(current.items()):
        if n <= LIMIT:
            continue
        allowed = baseline.get(rel)
        if allowed is None:
            errors.append(f"{rel}: {n} lines, over the {LIMIT}-line limit; split it")
        elif n > allowed:
            errors.append(f"{rel}: grew from {allowed} to {n} lines; it may shrink but never grow")
    return errors


def main(argv: list[str]) -> int:
    current = sources()
    baseline = read_baseline()
    if "--update" in argv:
        if "--accept-growth" in argv:
            write_baseline({rel: n for rel, n in current.items() if n > LIMIT})
        else:
            lowered = {
                rel: min(allowed, current[rel])
                for rel, allowed in baseline.items()
                if current.get(rel, 0) > LIMIT
            }
            write_baseline(lowered)
        print(f"baseline written: {BASELINE.name}")
        return 0
    errors = check(current, baseline)
    if errors:
        print("FAIL: source size ratchet")
        for err in errors:
            print(f"  {err}")
        return 1
    print(f"PASS: {len(current)} source files within the size ratchet ({len(baseline)} baselined)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
