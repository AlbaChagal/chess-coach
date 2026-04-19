from __future__ import annotations

from pathlib import Path

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import StructuredExplanation
from chesscoach.pipeline import (
    LOW_CONFIDENCE_WARNING,
    coaching_result_to_dict,
    complete_position,
    run_coaching_pipeline,
    run_explanation,
)
from chesscoach.pipeline_models import (
    AnalysisResult,
    CoachingRequest,
    CompletedPosition,
    ExplanationResult,
    ImageClick,
    PipelineWarning,
    VisionResult,
)

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
CLICK = ImageClick(x=123.0, y=456.0)


def make_request(**overrides) -> CoachingRequest:
    values = {
        "image": Path("board.jpg"),
        "side_to_move": "w",
        "white_king_start_click": CLICK,
        "castling_rights": None,
        "en_passant": None,
        "include_explanation": False,
        "explanation_provider": None,
        "explanation_model": None,
        "top_n": 3,
    }
    values.update(overrides)
    return CoachingRequest(**values)


def make_vision_result(**overrides) -> VisionResult:
    values = {
        "fen_placement": STARTING_PLACEMENT,
        "vision_confidence": 1.0,
        "orientation_status": "user_marked",
        "needs_user_confirmation": False,
        "white_king_start_click": CLICK,
        "debug": None,
    }
    values.update(overrides)
    return VisionResult(**values)


def make_position(**overrides) -> CompletedPosition:
    values = {
        "fen": f"{STARTING_PLACEMENT} w KQkq - 0 1",
        "fen_placement": STARTING_PLACEMENT,
        "side_to_move": "w",
        "castling_rights": "KQkq",
        "en_passant": "-",
        "source": "heuristic",
        "user_confirmed_orientation": True,
        "white_king_start_click": CLICK,
    }
    values.update(overrides)
    return CompletedPosition(**values)


def make_analysis_result(**overrides) -> AnalysisResult:
    moves = [
        MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3"]),
        MoveAnalysis("d4", "d2d4", 25, None, 20, ["d5"]),
    ]
    values = {
        "fen": f"{STARTING_PLACEMENT} w KQkq - 0 1",
        "top_moves": moves,
        "engine_depth": 20,
        "analysis_latency_ms": 12.5,
        "analysis_status": "success",
    }
    values.update(overrides)
    return AnalysisResult(**values)


def test_pipeline_requires_white_king_click() -> None:
    result = run_coaching_pipeline(
        make_request(white_king_start_click=None, side_to_move="w")
    )

    assert result.status == "partial"
    assert result.user_action_required == "white_king_start_click"
    assert result.analysis is None


def test_pipeline_requires_side_to_move_before_analysis(monkeypatch) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: STARTING_PLACEMENT,
    )

    result = run_coaching_pipeline(make_request(side_to_move=None))

    assert result.status == "partial"
    assert result.user_action_required == "side_to_move"
    assert result.position is None
    assert result.analysis is None


def test_pipeline_returns_failed_result_on_board_detection_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: (_ for _ in ()).throw(ValueError("bad image")),
    )

    result = run_coaching_pipeline(make_request())

    assert result.status == "failed"
    assert result.warnings == [LOW_CONFIDENCE_WARNING]
    assert result.vision.vision_confidence == 0.0


def test_complete_position_infers_full_castling_rights() -> None:
    position = complete_position(make_vision_result(), make_request())

    assert position is not None
    assert position.castling_rights == "KQkq"
    assert position.source == "heuristic"


def test_complete_position_infers_partial_castling_rights() -> None:
    placement = "4k3/8/8/8/8/8/8/R3K2R"

    position = complete_position(
        make_vision_result(fen_placement=placement),
        make_request(),
    )

    assert position is not None
    assert position.castling_rights == "KQ"


def test_complete_position_uses_explicit_castling_rights() -> None:
    position = complete_position(
        make_vision_result(),
        make_request(castling_rights="-"),
    )

    assert position is not None
    assert position.castling_rights == "-"
    assert position.source == "user"


def test_coaching_result_to_dict_includes_score_display(monkeypatch) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: STARTING_PLACEMENT,
    )
    monkeypatch.setattr(
        "chesscoach.pipeline.run_analysis",
        lambda position, top_n: make_analysis_result(),
    )
    result = run_coaching_pipeline(make_request())
    payload = coaching_result_to_dict(result)

    assert payload["analysis"]["top_moves"][0]["score_display"] == "+0.35"


def test_pipeline_success_path_runs_analysis_and_skips_explanation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: STARTING_PLACEMENT,
    )
    analysis = make_analysis_result()
    monkeypatch.setattr(
        "chesscoach.pipeline.run_analysis", lambda position, top_n: analysis
    )

    result = run_coaching_pipeline(make_request())

    assert result.status == "success"
    assert result.analysis == analysis
    assert result.explanation == ExplanationResult(
        move_uci=None,
        move_san=None,
        explanation_text=None,
        structured_explanation=None,
        provider=None,
        status="skipped",
    )


def test_run_explanation_skips_when_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline._pick_explanation_provider",
        lambda provider, model: (None, None, None),
    )
    monkeypatch.setattr(
        "chesscoach.pipeline.Explainer.analyze_position",
        lambda self, fen_before: self.analyze_move(fen_before, "e2e4"),
    )

    explanation, warnings = run_explanation(
        make_position(),
        make_analysis_result(),
        make_request(include_explanation=True),
    )

    assert explanation.status == "success"
    assert explanation.structured_explanation is not None
    assert warnings == []


def test_run_explanation_returns_text_when_provider_available(monkeypatch) -> None:
    class _Provider:
        def complete(self, system: str, user: str) -> str:
            return "Play e4 to control the center."

    class _Explainer:
        def __init__(self, engine, provider, top_n) -> None:
            self.engine = engine
            self.provider = provider
            self.top_n = top_n

        def analyze_position(self, fen_before: str):
            assert fen_before == make_position().fen
            return self.analyze_move(fen_before, "e2e4")

        def analyze_move(self, fen_before: str, move_uci: str):
            from chesscoach.explanation.models import ExplainedMove, MoveQuality

            assert move_uci == "e2e4"
            return ExplainedMove(
                fen_before=fen_before,
                move_played_san="e4",
                move_played_uci="e2e4",
                quality=MoveQuality(label="best", cp_loss=0, emoji=""),
                best_move=make_analysis_result().top_moves[0],
                alternatives=make_analysis_result().top_moves[1:],
                tactics_after_played=[],
                tactics_after_best=[],
            )

        def build_structured_explanation(self, explained):
            from chesscoach.explanation.models import StructuredExplanation

            return StructuredExplanation(
                summary="e4 takes the center.",
                what_the_move_does="It claims central space.",
                what_it_threatens="It opens lines for development.",
                why_it_is_best="It keeps the strongest evaluation.",
                why_alternatives_are_worse="The alternatives are less forcing.",
                alternatives=[],
                tactical_themes=[],
            )

        def narrate_explanation(self, explained, structured) -> str:
            return "Play e4 to control the center."

    monkeypatch.setattr(
        "chesscoach.pipeline._pick_explanation_provider",
        lambda provider, model: ("openai", _Provider(), None),
    )
    monkeypatch.setattr("chesscoach.pipeline.Explainer", _Explainer)

    explanation, warnings = run_explanation(
        make_position(),
        make_analysis_result(),
        make_request(include_explanation=True),
    )

    assert explanation.status == "success"
    assert explanation.provider == "openai"
    assert explanation.explanation_text is not None
    assert explanation.structured_explanation is not None
    assert "control the center" in explanation.explanation_text
    assert warnings == []


def test_run_explanation_returns_structured_only_on_ambiguous_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline._pick_explanation_provider",
        lambda provider, model: (
            None,
            None,
            PipelineWarning(
                code="explanation_skipped_ambiguous_provider",
                message="ambiguous",
            ),
        ),
    )

    explanation, warnings = run_explanation(
        make_position(),
        make_analysis_result(),
        make_request(include_explanation=True),
    )

    assert explanation.status == "success"
    assert explanation.explanation_text is None
    assert explanation.structured_explanation is not None
    assert warnings[0].code == "explanation_skipped_ambiguous_provider"


def test_run_explanation_keeps_structured_payload_when_text_generation_fails(
    monkeypatch,
) -> None:
    class _Provider:
        def complete(self, system: str, user: str) -> str:
            raise AssertionError("should not be called directly")

    class _Explainer:
        def __init__(self, engine, provider, top_n) -> None:
            self.provider = provider

        def analyze_position(self, fen_before: str):
            return type(
                "Explained",
                (),
                {"best_move": make_analysis_result().top_moves[0]},
            )()

        def build_structured_explanation(self, explained):
            from chesscoach.explanation.models import StructuredExplanation

            return StructuredExplanation(
                summary="e4 takes the center.",
                what_the_move_does="It claims central space.",
                what_it_threatens="It opens lines for development.",
                why_it_is_best="It keeps the strongest evaluation.",
                why_alternatives_are_worse="The alternatives are less forcing.",
                alternatives=[],
                tactical_themes=[],
            )

        def narrate_explanation(self, explained, structured) -> str:
            from chesscoach.explanation.models import ExplanationError

            raise ExplanationError("provider failed")

    monkeypatch.setattr(
        "chesscoach.pipeline._pick_explanation_provider",
        lambda provider, model: ("openai", _Provider(), None),
    )
    monkeypatch.setattr("chesscoach.pipeline.Explainer", _Explainer)

    explanation, warnings = run_explanation(
        make_position(),
        make_analysis_result(),
        make_request(include_explanation=True),
    )

    assert explanation.status == "success"
    assert explanation.structured_explanation is not None
    assert explanation.explanation_text is None
    assert warnings[0].code == "explanation_text_generation_failed"


def test_pipeline_json_includes_structured_explanation(monkeypatch) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: STARTING_PLACEMENT,
    )
    monkeypatch.setattr(
        "chesscoach.pipeline.run_analysis",
        lambda position, top_n: make_analysis_result(),
    )
    monkeypatch.setattr(
        "chesscoach.pipeline.run_explanation",
        lambda position, analysis, request: (
            ExplanationResult(
                move_uci="e2e4",
                move_san="e4",
                explanation_text="Play e4 to control the center.",
                structured_explanation=StructuredExplanation(
                    summary="e4 takes the center.",
                    what_the_move_does="It claims central space.",
                    what_it_threatens="It opens lines.",
                    why_it_is_best="It keeps the strongest evaluation.",
                    why_alternatives_are_worse="Alternatives are slower.",
                    alternatives=[],
                    tactical_themes=[],
                ),
                provider="openai",
                status="success",
            ),
            [],
        ),
    )

    result = run_coaching_pipeline(make_request(include_explanation=True))
    payload = coaching_result_to_dict(result)

    assert payload["explanation"]["structured_explanation"]["summary"] == (
        "e4 takes the center."
    )


def test_pipeline_does_not_require_real_engine_when_analysis_mocked(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chesscoach.pipeline.predict_fen",
        lambda image: STARTING_PLACEMENT,
    )
    monkeypatch.setattr(
        "chesscoach.pipeline.run_analysis",
        lambda position, top_n: make_analysis_result(),
    )

    result = run_coaching_pipeline(make_request())

    assert result.status == "success"
    assert result.position is not None
    assert result.analysis is not None
