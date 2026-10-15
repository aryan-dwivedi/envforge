"""Canonical JSON and content-hash ids."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str | bytes) -> str:
    data = s.encode("utf-8") if isinstance(s, str) else s
    return hashlib.sha256(data).hexdigest()


def short_id(prefix: str, *parts: str) -> str:
    return prefix + "_" + sha256_hex("\n".join(parts))[:16]
