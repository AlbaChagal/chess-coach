"""Tests for position-level line normalization helpers."""

from __future__ import annotations

import pytest

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import CandidateLine
from chesscoach.explanation.position_synthesizer import (
    candidate_line_has_aligned_continuations,
    normalize_move_analyses,
    normalize_move_analysis,
)


def test_normalize_move_analysis_preserves_root_and_continuations() -> None:
    move = MoveAnalysis(
        move_san="e4",
        move_uci="e2e4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation=["e5", "Nf3", "Nc6"],
        continuation_uci=["e7e5", "g1f3", "b8c6"],
    )

    line = normalize_move_analysis(move)

    assert isinstance(line, CandidateLine)
    assert line.root_move_san == "e4"
    assert line.root_move_uci == "e2e4"
    assert line.score_cp == 35
    assert line.depth == 20
    assert line.continuation_san == ["e5", "Nf3", "Nc6"]
    assert line.continuation_uci == ["e7e5", "g1f3", "b8c6"]


def test_normalize_move_analyses_preserves_input_order() -> None:
    moves = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5"], ["e7e5"]),
        MoveAnalysis("d4", "d2d4", 30, None, 20, ["d5"], ["d7d5"]),
    ]

    lines = normalize_move_analyses(moves)

    assert [line.root_move_uci for line in lines] == ["e2e4", "d2d4"]


def test_normalize_move_analyses_returns_empty_list_for_empty_input() -> None:
    assert normalize_move_analyses([]) == []


def test_normalize_move_analysis_allows_san_only_continuation() -> None:
    move = MoveAnalysis(
        move_san="e4",
        move_uci="e2e4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation=["e5", "Nf3"],
        continuation_uci=[],
    )

    line = normalize_move_analysis(move)

    assert line.continuation_san == ["e5", "Nf3"]
    assert line.continuation_uci == []


def test_normalize_move_analysis_allows_empty_continuation() -> None:
    move = MoveAnalysis("e4", "e2e4", 35, None, 20, [], [])

    line = normalize_move_analysis(move)

    assert line.continuation_san == []
    assert line.continuation_uci == []


def test_normalize_move_analysis_preserves_mate_score() -> None:
    move = MoveAnalysis("Qh7#", "h5h7", None, 1, 24, [], [])

    line = normalize_move_analysis(move)

    assert line.score_cp is None
    assert line.score_mate == 1


def test_candidate_line_has_aligned_continuations_when_lengths_match() -> None:
    line = CandidateLine(
        root_move_uci="e2e4",
        root_move_san="e4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=["e5", "Nf3"],
        continuation_uci=["e7e5", "g1f3"],
    )

    assert candidate_line_has_aligned_continuations(line) is True


def test_candidate_line_has_aligned_continuations_when_both_empty() -> None:
    line = CandidateLine(
        root_move_uci="e2e4",
        root_move_san="e4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=[],
        continuation_uci=[],
    )

    assert candidate_line_has_aligned_continuations(line) is True


def test_candidate_line_has_aligned_continuations_when_lengths_differ() -> None:
    line = CandidateLine(
        root_move_uci="e2e4",
        root_move_san="e4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=["e5", "Nf3"],
        continuation_uci=["e7e5"],
    )

    assert candidate_line_has_aligned_continuations(line) is False


@pytest.mark.parametrize(
    ("move_san", "move_uci", "message"),
    [
        ("?", "e2e4", "SAN"),
        ("e4", "?", "UCI"),
        ("", "e2e4", "SAN"),
        ("e4", "", "UCI"),
    ],
)
def test_normalize_move_analysis_rejects_missing_root_move(
    move_san: str,
    move_uci: str,
    message: str,
) -> None:
    move = MoveAnalysis(move_san, move_uci, 35, None, 20, [], [])

    with pytest.raises(ValueError, match=message):
        normalize_move_analysis(move)
