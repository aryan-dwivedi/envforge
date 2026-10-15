"""TRL dataset export and reward function."""

from __future__ import annotations

import json
from pathlib import Path

from envforge import __version__
from envforge.export.trl import export_trl, make_reward_func
from envforge.store import Store


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "data.jsonl").read_text().splitlines()]


def test_export_rows_and_manifest(accepted_store: tuple, tmp_path: Path) -> None:
    store: Store = accepted_store[0]
    tasks = store.load_tasks()
    split = tasks[0].split
    in_split = [t.task_id for t in tasks if t.split == split]
    result = export_trl(store, tmp_path / "out", split=split, instances_per_task=4)
    rows = _rows(tmp_path / "out")
    assert result["rows"] == len(rows) == 4 * len(in_split)
    assert result["gen_failures"] == 0
    assert {r["task_id"] for r in rows} == set(in_split)
    assert sorted({r["seed"] for r in rows}) == [10_000, 10_001, 10_002, 10_003]
    r = rows[0]
    assert r["prompt"][0]["role"] == "system" and r["prompt"][-1]["role"] == "user"
    assert json.loads(r["problem"])["prompt"] == r["prompt"][-1]["content"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest == {
        "store_dir": str(store.root.resolve()),
        "split": split,
        "tasks": in_split,
        "envforge_version": __version__,
    }

    other = next(s for s in ("train", "eval", "holdout") if s not in {t.split for t in tasks})
    empty = export_trl(store, tmp_path / "empty", split=other, instances_per_task=4)
    assert empty["rows"] == 0 and empty["tasks"] == []
    assert (tmp_path / "empty" / "data.jsonl").read_text() == ""


def test_reward_func(accepted_store: tuple, tmp_path: Path) -> None:
    store: Store = accepted_store[0]
    export_trl(store, tmp_path, split=store.load_tasks()[0].split, instances_per_task=2)
    rows = _rows(tmp_path)[:2]
    answers = [json.loads(r["problem"])["answer"] for r in rows]
    completions = [
        f"Thinking. <answer>{answers[0]}</answer>",
        [{"role": "assistant", "content": f"<answer>{answers[1]}</answer>"}],
        "<answer>not-a-number</answer>",
        [{"role": "assistant", "content": "no tag here"}],
    ]
    picks = [rows[0], rows[1], rows[0], rows[1]]
    n = len(picks)
    reward = make_reward_func(store.root)
    try:
        assert reward.__name__ == "envforge_reward"
        got = reward(
            prompts=[r["prompt"] for r in picks],
            completions=completions,
            task_id=[r["task_id"] for r in picks],
            problem=[r["problem"] for r in picks],
            seed=[r["seed"] for r in picks],
            trainer_state=None,
            log_extra=lambda **k: None,
            log_metric=lambda *a: None,
            completion_ids=[[1]] * n,
        )
        assert got == [1.0, 1.0, 0.0, 0.0]
        bad = reward(prompts=["p"], completions=["<answer>1</answer>"], task_id=["missing_task"], problem=["{}"])
        assert bad == [0.0]
    finally:
        reward.close()
