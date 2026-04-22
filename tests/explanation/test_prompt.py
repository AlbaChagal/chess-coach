"""Tests for the prompt builder."""

from __future__ import annotations

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import (
    CandidateLine,
    ExplainedMove,
    MoveQuality,
    PositionContext,
    StructuredPositionExplanation,
    StructuredPlayedMoveExplanation,
    StructuredExplanation,
    TacticInfo,
)
from chesscoach.explanation.prompt import (
    build_best_move_prompt,
    build_position_prompt,
    build_played_move_prompt,
)

_BEST = MoveAnalysis("Nf3", "g1f3", 35, None, 20, ["d5", "d4", "Nf6"])
_ALTERNATIVES = [MoveAnalysis("d4", "d2d4", 30, None, 20, [])]

_QUALITY_BLUNDER = MoveQuality(label="blunder", cp_loss=320, emoji="??")
_QUALITY_BEST = MoveQuality(label="best", cp_loss=0, emoji="")


def _make_explained(
    quality: MoveQuality = _QUALITY_BLUNDER,
    tactics_played: list[TacticInfo] | None = None,
    tactics_best: list[TacticInfo] | None = None,
) -> ExplainedMove:
    return ExplainedMove(
        fen_before="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        move_played_san="e5",
        move_played_uci="e7e5",
        quality=quality,
        best_move=_BEST,
        alternatives=_ALTERNATIVES,
        tactics_after_played=tactics_played or [],
        tactics_after_best=tactics_best or [],
    )


def _make_structured() -> StructuredExplanation:
    return StructuredExplanation(
        summary="Nf3 develops a piece and keeps the strongest setup.",
        what_the_move_does="It develops the knight and supports central control.",
        what_it_threatens="It increases central pressure and prepares quick castling.",
        why_it_is_best="It keeps the best engine evaluation on the board.",
        why_alternatives_are_worse="The alternatives are playable but slightly less precise.",
        alternatives=[],
        tactical_themes=["fork"],
        position_context=PositionContext(
            position_summary="White should finish development before expanding.",
            shared_plan="The strong lines aim to support f4 break.",
            what_all_good_lines_have_in_common=(
                "The good lines all support f4 break, even if they differ by move order."
            ),
            what_to_watch_out_for="Black's main idea is quick central counterplay.",
        ),
    )


def _make_position_structured() -> StructuredPositionExplanation:
    return StructuredPositionExplanation(
        position_summary="White is better and should build toward kingside expansion.",
        main_ideas=[
            "The strong lines repeatedly build toward f4 break.",
            "King safety appears before direct action.",
        ],
        shared_plan="The strong lines aim to support f4 break.",
        why_the_best_move_fits="The best move supports f4 break more directly.",
        what_all_good_lines_have_in_common=(
            "The good lines all support f4 break, even if they differ by move order."
        ),
        what_to_watch_out_for="Black's main idea is quick central counterplay.",
        candidate_move_roles=[
            "f4 is the most direct route toward f4 break.",
            "Nf3 supports the same plan more gradually.",
        ],
    )


def test_build_prompt_returns_tuple_of_strings() -> None:
    system, user = build_best_move_prompt(_make_explained(), _make_structured())
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_system_prompt_non_empty() -> None:
    system, _ = build_best_move_prompt(_make_explained(), _make_structured())
    assert len(system) > 0


def test_user_prompt_contains_fen() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())
    assert "rnbqkbnr" in user


def test_user_prompt_contains_move_san() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())
    assert "Nf3" in user


def test_user_prompt_contains_structured_summary() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())
    assert "keeps the strongest setup" in user


def test_user_prompt_contains_best_move_san() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())
    assert "Nf3" in user


def test_tactics_section_present_when_tactics_detected() -> None:
    tactic = TacticInfo(name="hanging_piece", description="Rook on e4 is hanging.")
    _, user = build_best_move_prompt(
        _make_explained(tactics_played=[tactic]),
        _make_structured(),
    )
    assert "Rook on e4 is hanging" in user


def test_tactics_section_none_when_empty() -> None:
    _, user = build_best_move_prompt(
        _make_explained(tactics_played=[]),
        _make_structured(),
    )
    assert "None detected" in user


def test_best_move_line_in_prompt() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())
    # Continuation moves should appear.
    assert "d5" in user


def test_prompt_contains_why_alternatives_are_worse() -> None:
    _, user = build_best_move_prompt(
        _make_explained(quality=_QUALITY_BEST),
        _make_structured(),
    )
    assert "alternatives are worse" in user.lower()


def test_best_move_prompt_includes_position_context() -> None:
    _, user = build_best_move_prompt(_make_explained(), _make_structured())

    assert "Position summary" in user
    assert "Shared plan" in user
    assert "What the good lines have in common" in user


def test_position_prompt_returns_tuple_of_strings() -> None:
    system, user = build_position_prompt(
        _make_position_structured(),
        fen_before=_make_explained().fen_before,
    )

    assert isinstance(system, str)
    assert isinstance(user, str)


def test_position_prompt_contains_structured_fields() -> None:
    _, user = build_position_prompt(
        _make_position_structured(),
        fen_before=_make_explained().fen_before,
    )

    assert "Position summary" in user
    assert "Main ideas" in user
    assert "Shared plan" in user
    assert "Candidate move roles" in user


def test_position_prompt_includes_candidate_lines_when_provided() -> None:
    candidate_lines = [
        CandidateLine("f2f4", "f4", 40, None, 20, [], []),
        CandidateLine("g1f3", "Nf3", 35, None, 20, [], []),
    ]
    _, user = build_position_prompt(
        _make_position_structured(),
        fen_before=_make_explained().fen_before,
        candidate_lines=candidate_lines,
    )

    assert "Candidate lines" in user
    assert "f4 (+0.40)" in user


def test_played_move_prompt_includes_position_context() -> None:
    structured = StructuredPlayedMoveExplanation(
        summary="e5 is a blunder compared to Nf3.",
        what_the_move_tried_to_do="It tries to grab space.",
        what_was_missed="It misses a stronger developing move.",
        what_changed_after_move="It creates tactical problems.",
        why_best_move_was_better="Nf3 keeps the position cleaner.",
        practical_lesson="Compare forcing replies before pushing pawns.",
        tactical_themes=["fork"],
        alternatives=[],
        position_context=PositionContext(
            position_summary="White should finish development before expanding.",
            shared_plan="The strong lines aim to support f4 break.",
            what_all_good_lines_have_in_common=(
                "The good lines all support f4 break, even if they differ by move order."
            ),
            what_to_watch_out_for="Black's main idea is quick central counterplay.",
        ),
    )

    _, user = build_played_move_prompt(_make_explained(), structured)

    assert "Position summary" in user
    assert "Shared plan" in user


def test_prompt_builders_handle_missing_position_context() -> None:
    structured = StructuredExplanation(
        summary="Nf3 develops a piece and keeps the strongest setup.",
        what_the_move_does="It develops the knight and supports central control.",
        what_it_threatens="It increases central pressure and prepares quick castling.",
        why_it_is_best="It keeps the best engine evaluation on the board.",
        why_alternatives_are_worse="The alternatives are playable but slightly less precise.",
        alternatives=[],
        tactical_themes=["fork"],
        position_context=None,
    )

    _, user = build_best_move_prompt(_make_explained(), structured)

    assert "Position summary" not in user


def test_played_move_prompt_contains_played_move_fields() -> None:
    structured = StructuredPlayedMoveExplanation(
        summary="e5 is a blunder compared to Nf3.",
        what_the_move_tried_to_do="It tries to grab space.",
        what_was_missed="It misses a stronger developing move.",
        what_changed_after_move="It creates tactical problems.",
        why_best_move_was_better="Nf3 keeps the position cleaner.",
        practical_lesson="Compare forcing replies before pushing pawns.",
        tactical_themes=["fork"],
        alternatives=[],
        position_context=None,
    )

    _, user = build_played_move_prompt(_make_explained(), structured)

    assert "Played move: e5" in user
    assert "What was missed" in user
    assert "Practical lesson" in user
