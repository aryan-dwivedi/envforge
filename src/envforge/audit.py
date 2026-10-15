"""Human audit of reward correctness: sample scored instances, ingest labels."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from envforge.answer import extract_answer, task_messages
from envforge.export.trl import MODULE, env_sandbox, generate_for_env, task_difficulty
from envforge.hashing import short_id
from envforge.llm import ChatModel
from envforge.procedural import scoring
from envforge.schemas import AuditResult, Budgets
from envforge.store import Store

AUDIT_SEED_BASE = 20_000
LABELS = frozenset({"correct", "incorrect"})


def _ask(policy: ChatModel, prompt: str) -> str:
    try:
        return policy.chat(task_messages(prompt), temperature=1.0, max_tokens=Budgets().max_tokens).text
    except Exception:
        return ""


def audit_sample(
    store: Store,
    policy: ChatModel,
    *,
    n: int = 200,
    out: Path,
    call_timeout_s: float = 5.0,
    concurrency: int = 8,
) -> int:
    """Write ``n`` unlabelled audit rows to ``out``; return the number written."""
    tasks = store.load_tasks()
    picks = [(tasks[j % len(tasks)], AUDIT_SEED_BASE + j) for j in range(n)] if tasks else []
    by_env: dict[str, list[int]] = {}
    for k, (t, _) in enumerate(picks):
        by_env.setdefault(t.env_id, []).append(k)
    problems: dict[int, dict[str, Any]] = {}
    for env_id, ks in by_env.items():
        pairs = [(picks[k][1], task_difficulty(picks[k][0])) for k in ks]
        for k, g in zip(ks, generate_for_env(store, env_id, pairs, call_timeout_s=call_timeout_s), strict=True):
            if g.problem is not None:
                problems[k] = g.problem
    ks = sorted(problems)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        responses = dict(zip(ks, pool.map(lambda k: _ask(policy, problems[k]["prompt"]), ks), strict=True))
    rewards: dict[int, float] = {}
    for env_id, env_ks in by_env.items():
        items = [k for k in env_ks if k in problems]
        if not items:
            continue
        sb = env_sandbox(store, env_id)
        try:
            fork = sb.fork()
            try:
                scores = scoring.score(
                    fork, MODULE, [(problems[k], responses[k]) for k in items], call_timeout_s=call_timeout_s
                )
            finally:
                fork.destroy()
        finally:
            sb.destroy()
        for k, s in zip(items, scores, strict=True):
            rewards[k] = s.value if s.error is None else 0.0
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for k in ks:
            t, seed = picks[k]
            row = {
                "audit_id": short_id("aud", t.task_id, str(seed)),
                "task_id": t.task_id,
                "seed": seed,
                "prompt": problems[k]["prompt"],
                "response": responses[k],
                "extracted_answer": extract_answer(responses[k]),
                "reference_answer": problems[k]["answer"],
                "reward": rewards.get(k, 0.0),
                "label": None,
                "auditor": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(ks)


def audit_ingest(store: Store, path: Path) -> dict:
    """Aggregate labelled rows into each task's ``report.human_audit``; return totals."""
    per_task: dict[str, dict[str, Any]] = {}
    rows = skipped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        r = json.loads(line)
        if r.get("label") not in LABELS or not r.get("auditor") or not r.get("task_id"):
            skipped += 1
            continue
        agg = per_task.setdefault(r["task_id"], {"n": 0, "fa": 0, "fr": 0, "auditors": set()})
        reward = float(r.get("reward") or 0.0)
        agg["n"] += 1
        agg["fa"] += int(reward == 1.0 and r["label"] == "incorrect")
        agg["fr"] += int(reward == 0.0 and r["label"] == "correct")
        agg["auditors"].add(str(r["auditor"]))
    now = datetime.now(UTC).isoformat()
    for task_id, agg in sorted(per_task.items()):
        report = store.load_report(task_id)
        report.human_audit = AuditResult(
            n_audited=agg["n"],
            false_accepts=agg["fa"],
            false_rejects=agg["fr"],
            auditor=",".join(sorted(agg["auditors"])),
            audited_at=now,
        )
        store.save_report(report)
    return {
        "rows": rows,
        "skipped": skipped,
        "n_audited": sum(a["n"] for a in per_task.values()),
        "false_accepts": sum(a["fa"] for a in per_task.values()),
        "false_rejects": sum(a["fr"] for a in per_task.values()),
    }
