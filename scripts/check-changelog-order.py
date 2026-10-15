#!/usr/bin/env python3
"""Assert CHANGELOG.md is newest-first.

Every `## ` heading must be `## [Unreleased]` or `## [X.Y.Z] - YYYY-MM-DD`.
`[Unreleased]` may appear once, above every release. Versions strictly
decrease and dates never increase top to bottom. The release workflow reads
the section for a tag from this file, so an entry out of place ships the
wrong notes.

    python3 scripts/check-changelog-order.py [PATH]   # default CHANGELOG.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

UNRELEASED = re.compile(r"^## \[Unreleased\]\s*$")
RELEASE = re.compile(r"^## \[(\d+)\.(\d+)\.(\d+)([^\]]*)\] - (\d{4}-\d{2}-\d{2})\s*$")


def check(text: str) -> tuple[list[str], int]:
    errors: list[str] = []
    headings = 0
    releases = 0
    prev_ver: tuple[int, int, int] | None = None
    prev_date: str | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("## "):
            continue
        headings += 1
        if UNRELEASED.match(line):
            if headings > 1:
                errors.append(f"{lineno}: [Unreleased] must be the first section")
            continue
        m = RELEASE.match(line)
        if not m:
            # A heading this parser cannot read would otherwise be skipped,
            # and a skipped heading is an unchecked one.
            errors.append(f"{lineno}: heading is neither [Unreleased] nor [X.Y.Z] - YYYY-MM-DD: {line!r}")
            continue
        releases += 1
        ver = (int(m[1]), int(m[2]), int(m[3]))
        date = m[5]
        if prev_ver is not None and ver >= prev_ver:
            cur, above = ".".join(map(str, ver)), ".".join(map(str, prev_ver))
            errors.append(f"{lineno}: version {cur} is not below {above} above it")
        if prev_date is not None and date > prev_date:
            errors.append(f"{lineno}: date {date} is newer than {prev_date} above it")
        prev_ver, prev_date = ver, date
    if headings == 0:
        errors.append("no ## sections parsed; the heading format changed and this check stopped measuring anything")
    return errors, releases


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    errors, releases = check(path.read_text())
    if errors:
        for err in errors:
            print(f"FAIL:{path.name}:{err}")
        return 1
    print(f"PASS: {path.name} is newest-first ({releases} releases)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
