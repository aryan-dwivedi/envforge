"""G10: near-duplicate and eval-contamination checks for procedural environments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from datasketch import MinHash

from envforge.config import ConfigError, SuiteConfig
from envforge.llm import EmbedModel
from envforge.schemas import GateResult

EMBED_SIM_FAIL = 0.92
JACCARD_FAIL = 0.8
NUM_PERM = 128
MINHASH_SEED = 1
SCHEME = "affine32"
SHINGLE = 5
NGRAM = 13


def dedup_text(description: str, prompts: list[str]) -> str:
    """Text embedded and MinHashed for near-duplicate detection."""
    return description + "\n\n" + "\n\n".join(prompts)


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize ``vec``; a zero vector raises ``ValueError``."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in vec]


def minhash_of(text: str) -> MinHash:
    """MinHash over lowercase word 5-shingles of ``text``."""
    words = text.lower().split()
    m = MinHash(num_perm=NUM_PERM, seed=MINHASH_SEED, scheme=SCHEME)
    if len(words) < SHINGLE:
        m.update(" ".join(words).encode())
        return m
    for i in range(len(words) - SHINGLE + 1):
        m.update(" ".join(words[i : i + SHINGLE]).encode())
    return m


def minhash_from_values(values: list[int]) -> MinHash:
    """Rebuild a stored MinHash from its hash values."""
    return MinHash(num_perm=NUM_PERM, seed=MINHASH_SEED, hashvalues=values, scheme=SCHEME)


def _ngram_digests(text: str) -> list[bytes]:
    toks = text.lower().split()
    return [
        hashlib.blake2b(" ".join(toks[i : i + NGRAM]).encode(), digest_size=8).digest()
        for i in range(len(toks) - NGRAM + 1)
    ]


@dataclass(frozen=True)
class DedupRecord:
    env_id: str
    embedding: list[float]
    minhash: list[int]


@dataclass(frozen=True)
class DedupResult:
    gate: GateResult
    near_dup_max_sim: float
    max_jaccard: float
    contamination: dict[str, float]
    embedding: list[float]
    minhash: list[int]


class ContaminationIndex:
    """Per-suite sets of 13-gram digests from eval suite JSONL files."""

    def __init__(self, suites: dict[str, set[bytes]]) -> None:
        self.suites = suites

    @classmethod
    def from_suites(cls, suites: list[SuiteConfig]) -> ContaminationIndex:
        out: dict[str, set[bytes]] = {}
        for s in suites:
            try:
                lines = s.path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as e:
                raise ConfigError(f"eval suite {s.name}: cannot read {s.path}: {e}") from e
            grams: set[bytes] = set()
            for no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ConfigError(f"eval suite {s.name}: {s.path}:{no}: invalid JSON: {e}") from e
                if not isinstance(row, dict) or not isinstance(row.get(s.field), str):
                    raise ConfigError(f"eval suite {s.name}: {s.path}:{no}: missing string field {s.field!r}")
                grams.update(_ngram_digests(row[s.field]))
            out[s.name] = grams
        return cls(out)

    def rates(self, prompts: list[str]) -> dict[str, float]:
        """Per suite: fraction of the prompts' 13-grams found in that suite."""
        grams = [g for p in prompts for g in _ngram_digests(p)]
        total = len(grams)
        return {
            name: (sum(1 for g in grams if g in suite) / total if total else 0.0)
            for name, suite in self.suites.items()
        }


def _similarity(
    embedding: list[float], minhash: list[int], accepted: list[DedupRecord]
) -> tuple[float, str | None, float, str | None]:
    best_sim, sim_id, best_j, j_id = 0.0, None, 0.0, None
    mine = minhash_from_values(minhash)
    for rec in accepted:
        sim = sum(a * b for a, b in zip(embedding, rec.embedding, strict=True))
        if sim_id is None or sim > best_sim:
            best_sim, sim_id = sim, rec.env_id
        jac = mine.jaccard(minhash_from_values(rec.minhash))
        if j_id is None or jac > best_j:
            best_j, j_id = jac, rec.env_id
    return max(best_sim, 0.0) if accepted else 0.0, sim_id, best_j, j_id


def _sim_failures(sim: float, sim_id: str | None, jac: float, j_id: str | None) -> list[str]:
    fails = []
    if sim >= EMBED_SIM_FAIL:
        fails.append(f"near-duplicate of {sim_id}: embedding cosine {sim:.3f} >= {EMBED_SIM_FAIL}")
    if jac >= JACCARD_FAIL:
        fails.append(f"near-duplicate of {j_id}: minhash jaccard {jac:.3f} >= {JACCARD_FAIL}")
    return fails


def recheck(embedding: list[float], minhash: list[int], accepted: list[DedupRecord]) -> GateResult:
    """Similarity half of G10, re-run under the store lock before packaging."""
    sim, sim_id, jac, j_id = _similarity(embedding, minhash, accepted)
    fails = _sim_failures(sim, sim_id, jac, j_id)
    return GateResult(
        passed=not fails,
        metrics={"near_dup_max_sim": sim, "max_jaccard": jac},
        detail="; ".join(fails) if fails else "no near-duplicates",
    )


def g10(
    *,
    description: str,
    dedup_prompts: list[str],
    probe_prompts: list[str],
    embedder: EmbedModel,
    accepted: list[DedupRecord],
    index: ContaminationIndex,
) -> DedupResult:
    text = dedup_text(description, dedup_prompts)
    embedding = normalize(embedder.embed([text])[0])
    minhash = [int(v) for v in minhash_of(text).hashvalues]
    sim, sim_id, jac, j_id = _similarity(embedding, minhash, accepted)
    fails = _sim_failures(sim, sim_id, jac, j_id)
    contamination = index.rates(probe_prompts)
    for name, rate in contamination.items():
        if rate > 0:
            fails.append(f"contaminated by eval suite {name}: 13-gram overlap {rate:.3f}")
    cmax = max(contamination.values(), default=0.0)
    if fails:
        detail = "; ".join(fails)
    else:
        detail = "no near-duplicates or contamination"
        if not contamination:
            detail += "; no eval suites registered"
    gate = GateResult(
        passed=not fails,
        metrics={"near_dup_max_sim": sim, "max_jaccard": jac, "contamination_max": cmax},
        detail=detail,
    )
    return DedupResult(gate, sim, jac, contamination, embedding, minhash)
