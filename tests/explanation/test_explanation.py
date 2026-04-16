"""Tests for the PositionExplainer stub (backward-compat) and Explainer integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation import Explainer, PositionExplainer

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SAMPLE_MOVES = [
    MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3", "Nc6"]),
]


# ---------------------------------------------------------------------------
# PositionExplainer stub — backward compatibility
# ---------------------------------------------------------------------------


def test_position_explainer_stub_raises_not_implemented() -> None:
    explainer = PositionExplainer()
    with pytest.raises(NotImplementedError):
        explainer.explain(STARTING_FEN, SAMPLE_MOVES)


# ---------------------------------------------------------------------------
# Explainer — wiring tests with mocked engine + provider
# ---------------------------------------------------------------------------


def _make_explainer(provider_text: str = "Good move!") -> Explainer:
    """Build an Explainer with fully mocked engine and provider."""
    # Engine returns the same analysis for any board (before and after move).
    engine = MagicMock()
    engine.get_best_moves.return_value = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3"]),
        MoveAnalysis("d4", "d2d4", 30, None, 20, ["d5"]),
    ]

    provider = MagicMock()
    provider.complete.return_value = provider_text

    return Explainer(engine, provider)


def test_explain_move_returns_string() -> None:
    explainer = _make_explainer("Nice!")
    result = explainer.explain_move(STARTING_FEN, "e2e4")
    assert isinstance(result, str)
    assert len(result) > 0


def test_explain_move_calls_provider_once() -> None:
    engine = MagicMock()
    engine.get_best_moves.return_value = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, []),
    ]
    provider = MagicMock()
    provider.complete.return_value = "text"

    explainer = Explainer(engine, provider)
    explainer.explain_move(STARTING_FEN, "e2e4")
    provider.complete.assert_called_once()


def test_explain_move_provider_receives_non_empty_prompts() -> None:
    engine = MagicMock()
    engine.get_best_moves.return_value = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, []),
    ]
    provider = MagicMock()
    provider.complete.return_value = "ok"

    explainer = Explainer(engine, provider)
    explainer.explain_move(STARTING_FEN, "e2e4")

    call_args = provider.complete.call_args
    system, user = call_args.args
    assert len(system) > 0
    assert len(user) > 0


def test_analyze_move_returns_explained_move() -> None:
    from chesscoach.explanation.models import ExplainedMove

    explainer = _make_explainer()
    result = explainer.analyze_move(STARTING_FEN, "e2e4")
    assert isinstance(result, ExplainedMove)


def test_analyze_move_move_san_correct() -> None:
    explainer = _make_explainer()
    result = explainer.analyze_move(STARTING_FEN, "e2e4")
    assert result.move_played_san == "e4"
    assert result.move_played_uci == "e2e4"


def test_analyze_move_does_not_call_provider() -> None:
    engine = MagicMock()
    engine.get_best_moves.return_value = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, []),
    ]
    provider = MagicMock()

    explainer = Explainer(engine, provider)
    explainer.analyze_move(STARTING_FEN, "e2e4")
    provider.complete.assert_not_called()


def test_explain_legacy_interface_works() -> None:
    explainer = _make_explainer("Legacy works!")
    result = explainer.explain(STARTING_FEN, SAMPLE_MOVES)
    assert isinstance(result, str)


def test_explain_legacy_raises_on_empty_moves() -> None:
    explainer = _make_explainer()
    with pytest.raises(ValueError, match="No moves"):
        explainer.explain(STARTING_FEN, [])


def test_analyze_move_invalid_fen_raises() -> None:
    explainer = _make_explainer()
    with pytest.raises(ValueError, match="Invalid FEN"):
        explainer.analyze_move("not_a_fen", "e2e4")


def test_analyze_move_illegal_move_raises() -> None:
    explainer = _make_explainer()
    with pytest.raises(ValueError):
        explainer.analyze_move(STARTING_FEN, "e2e5")  # illegal pawn jump
