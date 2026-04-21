"""Tests for position-level line normalization helpers."""

from __future__ import annotations

import pytest

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import CandidateLine, LineFeature
from chesscoach.explanation.position_synthesizer import (
    candidate_line_has_aligned_continuations,
    extract_line_features,
    extract_line_features_for_lines,
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


def test_extract_line_features_detects_root_pawn_break() -> None:
    line = CandidateLine(
        root_move_uci="f2f4",
        root_move_san="f4",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=[],
        continuation_uci=[],
    )

    features = extract_line_features(line)

    assert _feature_labels(features, "pawn_break") == ["f4 break"]
    assert features[0].ply_index == 0
    assert features[0].move_uci == "f2f4"


def test_extract_line_features_detects_continuation_pawn_break() -> None:
    line = CandidateLine(
        root_move_uci="g1f3",
        root_move_san="Nf3",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=["d5", "f4"],
        continuation_uci=["d7d5", "f2f4"],
    )

    features = extract_line_features(line)

    pawn_breaks = [feature for feature in features if feature.kind == "pawn_break"]
    assert pawn_breaks[-1].label == "f4 break"
    assert pawn_breaks[-1].ply_index == 2
    assert pawn_breaks[-1].move_uci == "f2f4"


def test_extract_line_features_detects_castling() -> None:
    line = CandidateLine(
        root_move_uci="e1g1",
        root_move_san="O-O",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=[],
        continuation_uci=[],
    )

    features = extract_line_features(line)

    assert _feature_labels(features, "king_safety") == ["kingside castling"]


def test_extract_line_features_detects_piece_improvement() -> None:
    line = CandidateLine(
        root_move_uci="g1f3",
        root_move_san="Nf3",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=[],
        continuation_uci=[],
    )

    features = extract_line_features(line)

    assert _feature_labels(features, "piece_improvement") == ["Nf3 development"]


def test_extract_line_features_detects_tactical_motif_from_check() -> None:
    line = CandidateLine(
        root_move_uci="h5h7",
        root_move_san="Qh7+",
        score_cp=None,
        score_mate=3,
        depth=24,
        continuation_san=[],
        continuation_uci=[],
    )

    features = extract_line_features(line)

    assert _feature_labels(features, "tactical_motif") == ["check"]


def test_extract_line_features_collects_multiple_feature_families() -> None:
    line = CandidateLine(
        root_move_uci="g1f3",
        root_move_san="Nf3",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=["O-O", "f4"],
        continuation_uci=["e8g8", "f2f4"],
    )

    features = extract_line_features(line)

    assert _feature_labels(features, "piece_improvement") == ["Nf3 development"]
    assert _feature_labels(features, "king_safety") == ["kingside castling"]
    assert _feature_labels(features, "pawn_break") == ["f4 break"]


def test_extract_line_features_uses_none_for_unaligned_continuation_uci() -> None:
    line = CandidateLine(
        root_move_uci="g1f3",
        root_move_san="Nf3",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=["O-O", "f4"],
        continuation_uci=["e8g8"],
    )

    features = extract_line_features(line)

    continuation_feature = next(
        feature
        for feature in features
        if feature.ply_index == 1 and feature.kind == "king_safety"
    )
    assert continuation_feature.move_uci is None


def test_extract_line_features_allows_empty_continuation() -> None:
    line = CandidateLine(
        root_move_uci="g1f3",
        root_move_san="Nf3",
        score_cp=35,
        score_mate=None,
        depth=20,
        continuation_san=[],
        continuation_uci=[],
    )

    features = extract_line_features(line)

    assert len(features) == 1
    assert features[0].label == "Nf3 development"


def test_extract_line_features_for_lines_preserves_line_order() -> None:
    lines = [
        CandidateLine("g1f3", "Nf3", 35, None, 20, ["O-O"], ["e8g8"]),
        CandidateLine("f2f4", "f4", 30, None, 20, [], []),
    ]

    features_by_line = extract_line_features_for_lines(lines)

    assert len(features_by_line) == 2
    assert _feature_labels(features_by_line[0], "piece_improvement") == [
        "Nf3 development"
    ]
    assert _feature_labels(features_by_line[1], "pawn_break") == ["f4 break"]


def _feature_labels(features: list[LineFeature], kind: str) -> list[str]:
    return [feature.label for feature in features if feature.kind == kind]
