"""Tests for the prompt builder."""

from __future__ import annotations

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import ExplainedMove, MoveQuality, TacticInfo
from chesscoach.explanation.prompt import build_prompt

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


def test_build_prompt_returns_tuple_of_strings() -> None:
    system, user = build_prompt(_make_explained())
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_system_prompt_non_empty() -> None:
    system, _ = build_prompt(_make_explained())
    assert len(system) > 0


def test_user_prompt_contains_fen() -> None:
    _, user = build_prompt(_make_explained())
    assert "rnbqkbnr" in user


def test_user_prompt_contains_move_san() -> None:
    _, user = build_prompt(_make_explained())
    assert "e5" in user


def test_user_prompt_contains_quality_label() -> None:
    _, user = build_prompt(_make_explained())
    assert "blunder" in user.lower()


def test_user_prompt_contains_best_move_san() -> None:
    _, user = build_prompt(_make_explained())
    assert "Nf3" in user


def test_tactics_section_present_when_tactics_detected() -> None:
    tactic = TacticInfo(name="hanging_piece", description="Rook on e4 is hanging.")
    _, user = build_prompt(_make_explained(tactics_played=[tactic]))
    assert "Rook on e4 is hanging" in user


def test_tactics_section_none_when_empty() -> None:
    _, user = build_prompt(_make_explained(tactics_played=[]))
    assert "None detected" in user


def test_best_move_line_in_prompt() -> None:
    _, user = build_prompt(_make_explained())
    # Continuation moves should appear.
    assert "d5" in user


def test_best_quality_label_in_prompt() -> None:
    _, user = build_prompt(_make_explained(quality=_QUALITY_BEST))
    assert "best" in user.lower()
