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


def test_analyze_position_returns_best_move_analysis() -> None:
    explainer = _make_explainer()
    result = explainer.analyze_position(STARTING_FEN)

    assert result.best_move.move_uci == "e2e4"
    assert result.move_played_uci == "e2e4"


def test_build_structured_explanation_returns_typed_payload() -> None:
    from chesscoach.explanation.models import StructuredExplanation

    explainer = _make_explainer()
    explained = explainer.analyze_position(STARTING_FEN)
    structured = explainer.build_structured_explanation(explained)

    assert isinstance(structured, StructuredExplanation)
    assert structured.summary
    assert structured.what_the_move_does
    assert structured.why_it_is_best


def test_build_structured_explanation_includes_alternatives() -> None:
    explainer = _make_explainer()
    explained = explainer.analyze_position(STARTING_FEN)
    structured = explainer.build_structured_explanation(explained)

    assert structured.alternatives
    assert structured.alternatives[0].move_san == "d4"


def test_build_structured_explanation_includes_position_context() -> None:
    explainer = _make_explainer()
    explained = explainer.analyze_position(STARTING_FEN)

    structured = explainer.build_structured_explanation(explained)

    assert structured.position_context is not None
    assert structured.position_context.position_summary
    assert structured.position_context.shared_plan
    assert "broader plan" in structured.why_it_is_best


def test_build_played_move_result_returns_quality_fields() -> None:
    explainer = _make_explainer()
    explained = explainer.analyze_move(STARTING_FEN, "e2e4")
    played = explainer.build_played_move_result(explained)

    assert played.move_uci == "e2e4"
    assert played.quality_label == explained.quality.label
    assert played.cp_loss == explained.quality.cp_loss


def test_build_best_move_comparison_returns_gap_text() -> None:
    explainer = _make_explainer()
    explained = explainer.analyze_move(STARTING_FEN, "d2d4")
    comparison = explainer.build_best_move_comparison(explained)

    assert comparison.best_move_san == "e4"
    assert comparison.played_move_san == "d4"
    assert comparison.why_best_move_is_better


def test_build_structured_played_move_explanation_returns_typed_payload() -> None:
    from chesscoach.explanation.models import StructuredPlayedMoveExplanation

    explainer = _make_explainer()
    explained = explainer.analyze_move(STARTING_FEN, "d2d4")
    structured = explainer.build_structured_played_move_explanation(explained)

    assert isinstance(structured, StructuredPlayedMoveExplanation)
    assert structured.summary
    assert structured.practical_lesson


def test_build_structured_played_move_explanation_includes_position_context() -> None:
    explainer = _make_explainer()
    explained = explainer.analyze_move(STARTING_FEN, "d2d4")

    structured = explainer.build_structured_played_move_explanation(explained)

    assert structured.position_context is not None
    assert structured.position_context.what_all_good_lines_have_in_common
    assert "shared plan" in structured.what_was_missed


def test_narrate_played_move_explanation_uses_provider_once() -> None:
    provider = MagicMock()
    provider.complete.return_value = "You missed a cleaner move."
    explainer = _make_explainer()
    explainer._provider = provider
    explained = explainer.analyze_move(STARTING_FEN, "d2d4")
    structured = explainer.build_structured_played_move_explanation(explained)

    text = explainer.narrate_played_move_explanation(explained, structured)

    assert text == "You missed a cleaner move."
    provider.complete.assert_called_once()


def test_narrate_explanation_uses_structured_input() -> None:
    engine = MagicMock()
    engine.get_best_moves.return_value = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, []),
        MoveAnalysis("d4", "d2d4", 20, None, 20, []),
    ]
    provider = MagicMock()
    provider.complete.return_value = "Strong center control."

    explainer = Explainer(engine, provider)
    explained = explainer.analyze_position(STARTING_FEN)
    structured = explainer.build_structured_explanation(explained)

    text = explainer.narrate_explanation(explained, structured)

    assert text == "Strong center control."
    provider.complete.assert_called_once()


def test_analyze_position_theme_returns_position_theme() -> None:
    from chesscoach.explanation.models import PositionTheme

    explainer = _make_explainer()

    theme = explainer.analyze_position_theme(STARTING_FEN)

    assert isinstance(theme, PositionTheme)
    assert theme.summary


def test_build_structured_position_explanation_returns_typed_payload() -> None:
    from chesscoach.explanation.models import StructuredPositionExplanation

    explainer = _make_explainer()

    structured = explainer.build_structured_position_explanation(STARTING_FEN)

    assert isinstance(structured, StructuredPositionExplanation)
    assert structured.position_summary
    assert structured.candidate_move_roles


def test_analyze_position_reuses_prefetched_engine_moves() -> None:
    engine = MagicMock()
    engine.get_best_moves.side_effect = [
        [
            MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5"], ["e7e5"]),
            MoveAnalysis("d4", "d2d4", 30, None, 20, ["d5"], ["d7d5"]),
        ],
        [
            MoveAnalysis("e5", "e7e5", -20, None, 20, ["Nf3"], ["g1f3"]),
        ],
    ]
    explainer = Explainer(engine, None)

    result = explainer.analyze_position(STARTING_FEN)

    assert result.best_move.move_uci == "e2e4"
    assert engine.get_best_moves.call_count == 2


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
