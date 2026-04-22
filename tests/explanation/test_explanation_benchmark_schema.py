"""Tests for explanation benchmark dataset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.explanation.dataset import load_jsonl


def test_load_jsonl_reads_benchmark_cases(tmp_path: Path) -> None:
    dataset_path = tmp_path / "positions.jsonl"
    dataset_path.write_text(
        '{"id":"case-1","fen":"8/8/8/8/8/8/8/8 w - - 0 1","top_n":3,'
        '"expected":{"shared_ideas":[],"divergence":"distinct_plans",'
        '"best_move_role_contains":[],"shared_plan_contains":[],'
        '"counterplay_contains":[],"counterplay_fallback_ok":true}}\n'
    )

    cases = load_jsonl(dataset_path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].expected.divergence == "distinct_plans"


def test_load_jsonl_rejects_missing_expected_object(tmp_path: Path) -> None:
    dataset_path = tmp_path / "positions.jsonl"
    dataset_path.write_text(
        '{"id":"case-1","fen":"8/8/8/8/8/8/8/8 w - - 0 1","top_n":3}\n'
    )

    with pytest.raises(ValueError, match="expected object"):
        load_jsonl(dataset_path)


def test_load_jsonl_rejects_non_string_list_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "positions.jsonl"
    dataset_path.write_text(
        '{"id":"case-1","fen":"8/8/8/8/8/8/8/8 w - - 0 1","top_n":3,'
        '"expected":{"shared_ideas":[],"divergence":"distinct_plans",'
        '"best_move_role_contains":[1],"shared_plan_contains":[],'
        '"counterplay_contains":[],"counterplay_fallback_ok":true}}\n'
    )

    with pytest.raises(ValueError, match="best_move_role_contains"):
        load_jsonl(dataset_path)
