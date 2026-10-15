"""Audit sampling and label ingestion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from conftest import ScriptedLLM, answer_rule

from envforge.answer import extract_answer
from envforge.audit import audit_ingest, audit_sample
from envforge.store import Store


def test_audit_sample(accepted_store: tuple, tmp_path: Path) -> None:
    store: Store = accepted_store[0]
    out = tmp_path / "audit.jsonl"
    n = audit_sample(store, ScriptedLLM("policy-model", answer_rule(2)), n=12, out=out, concurrency=4)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert n == len(rows) == 12
    assert all(r["seed"] >= 20_000 for r in rows)
    assert len({r["audit_id"] for r in rows}) == 12
    for r in rows:
        assert r["label"] is None and r["auditor"] is None
        assert r["extracted_answer"] == extract_answer(r["response"])
        assert r["reward"] == (1.0 if r["extracted_answer"] == r["reference_answer"] else 0.0)
    assert {r["reward"] for r in rows} == {0.0, 1.0}


def test_audit_policy_error(accepted_store: tuple, tmp_path: Path) -> None:
    class Broken:
        model = "broken"

        def chat(self, messages: list[dict], **kwargs: object) -> object:
            raise RuntimeError("down")

    out = tmp_path / "audit.jsonl"
    assert audit_sample(accepted_store[0], Broken(), n=2, out=out) == 2  # type: ignore[arg-type]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["response"], r["reward"]) for r in rows] == [("", 0.0), ("", 0.0)]


def test_audit_ingest(accepted_store: tuple, tmp_path: Path) -> None:
    src: Store = accepted_store[0]
    shutil.copytree(src.root, tmp_path / "store")
    store = Store(tmp_path / "store")
    task_id = store.load_tasks()[0].task_id
    base = {"task_id": task_id, "prompt": "p", "response": "r", "extracted_answer": "1", "reference_answer": "1"}
    rows = [
        {**base, "seed": 20_000, "reward": 1.0, "label": "correct", "auditor": "b"},
        {**base, "seed": 20_001, "reward": 1.0, "label": "incorrect", "auditor": "a"},
        {**base, "seed": 20_002, "reward": 0.0, "label": "correct", "auditor": "a"},
        {**base, "seed": 20_003, "reward": 0.0, "label": "incorrect", "auditor": "b"},
        {**base, "seed": 20_004, "reward": 0.0, "label": None, "auditor": None},
        {**base, "seed": 20_005, "reward": 1.0, "label": "unsure", "auditor": "a"},
        {**base, "seed": 20_006, "reward": 1.0, "label": "correct", "auditor": None},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    totals = audit_ingest(store, path)
    assert totals == {"rows": 7, "skipped": 3, "n_audited": 4, "false_accepts": 1, "false_rejects": 1}
    audit = store.load_report(task_id).human_audit
    assert audit is not None
    assert (audit.n_audited, audit.false_accepts, audit.false_rejects, audit.auditor) == (4, 1, 1, "a,b")
    assert src.load_report(task_id).human_audit is None
