"""Tests for position-level explanation models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from chesscoach.explanation.models import (
    CandidateLine,
    LineFeature,
    PositionTheme,
    RecurringIdea,
    StructuredPositionExplanation,
)


def test_candidate_line_stores_normalized_engine_fields() -> None:
    line = CandidateLine(
        root_move_uci="e2e4",
        root_move_san="e4",
        score_cp=42,
        score_mate=None,
        depth=20,
        continuation_san=["e5", "Nf3", "Nc6"],
        continuation_uci=["e7e5", "g1f3", "b8c6"],
    )

    assert line.root_move_uci == "e2e4"
    assert line.root_move_san == "e4"
    assert line.score_cp == 42
    assert line.depth == 20
    assert line.continuation_san[0] == "e5"
    assert line.continuation_uci[-1] == "b8c6"


def test_line_feature_captures_single_line_signal() -> None:
    feature = LineFeature(
        kind="pawn_break",
        label="f4 break",
        ply_index=3,
        move_uci="f3f4",
        move_san="f4",
        description="White prepares and achieves the f4 pawn break.",
    )

    assert feature.kind == "pawn_break"
    assert feature.label == "f4 break"
    assert feature.ply_index == 3
    assert feature.move_uci == "f3f4"


def test_recurring_idea_tracks_cross_line_support() -> None:
    idea = RecurringIdea(
        kind="pawn_break",
        label="f4 break",
        evidence_lines=[0, 1, 2],
        support=1.0,
        description="The top lines all converge on the kingside pawn break.",
    )

    assert idea.evidence_lines == [0, 1, 2]
    assert idea.support == 1.0


def test_position_theme_allows_partial_optional_fields() -> None:
    theme = PositionTheme(
        summary="White should improve king safety before opening the kingside.",
        recurring_ideas=[],
        side_to_move_plan="Finish development and prepare f4.",
        opponent_counterplay="Black wants ...c5 and queenside pressure.",
        critical_decision=None,
        best_move_role="The best move keeps the plan flexible.",
        line_divergence_summary="The top lines share a plan and differ by move order.",
    )

    assert theme.summary.startswith("White should")
    assert theme.recurring_ideas == []
    assert theme.critical_decision is None


def test_structured_position_explanation_holds_product_fields() -> None:
    explanation = StructuredPositionExplanation(
        position_summary="White is better and should prepare a kingside expansion.",
        main_ideas=[
            "Prepare the f4 break.",
            "Keep the king safe before opening the position.",
        ],
        shared_plan="The good lines all build toward kingside expansion.",
        why_the_best_move_fits="It develops while keeping the expansion plan intact.",
        what_all_good_lines_have_in_common=(
            "They all improve coordination before committing to f4."
        ),
        what_to_watch_out_for="Black's main counterplay is ...c5.",
        candidate_move_roles=[
            "Nf3 develops and supports the center.",
            "h3 prepares the same plan more slowly.",
        ],
    )

    assert len(explanation.main_ideas) == 2
    assert explanation.candidate_move_roles[0].startswith("Nf3")


def test_position_models_are_frozen() -> None:
    line = CandidateLine(
        root_move_uci="e2e4",
        root_move_san="e4",
        score_cp=42,
        score_mate=None,
        depth=20,
        continuation_san=["e5"],
        continuation_uci=["e7e5"],
    )

    with pytest.raises(FrozenInstanceError):
        setattr(line, "depth", 22)
