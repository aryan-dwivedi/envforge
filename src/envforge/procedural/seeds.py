"""Built-in seed catalogue for the procedural environment family."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Seed:
    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    license: str
    source: str


def load_seeds() -> list[Seed]:
    """Load the packaged `seeds.toml` in file order; duplicate ids raise ValueError."""
    raw = tomllib.loads((files("envforge.procedural") / "seeds.toml").read_text(encoding="utf-8"))
    seeds: list[Seed] = []
    seen: set[str] = set()
    for entry in raw.get("seed", []):
        seed = Seed(
            id=entry["id"],
            title=entry["title"],
            description=entry["description"],
            tags=tuple(entry["tags"]),
            license=entry["license"],
            source=entry["source"],
        )
        if seed.id in seen:
            raise ValueError(f"duplicate seed id: {seed.id}")
        seen.add(seed.id)
        seeds.append(seed)
    return seeds
