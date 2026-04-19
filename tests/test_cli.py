from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from chesscoach.pipeline_models import (
    AnalysisResult,
    CoachingResult,
    CompletedPosition,
    ExplanationResult,
    ImageClick,
    PipelineWarning,
    VisionResult,
)
from chesscoach.explanation.models import StructuredExplanation

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def make_pipeline_result(**overrides) -> CoachingResult:
    result = CoachingResult(
        vision=VisionResult(
            fen_placement=STARTING_PLACEMENT,
            vision_confidence=1.0,
            orientation_status="user_marked",
            needs_user_confirmation=False,
            white_king_start_click=ImageClick(1.0, 2.0),
        ),
        position=CompletedPosition(
            fen=f"{STARTING_PLACEMENT} w KQkq - 0 1",
            fen_placement=STARTING_PLACEMENT,
            side_to_move="w",
            castling_rights="KQkq",
            en_passant="-",
            source="heuristic",
            user_confirmed_orientation=True,
            white_king_start_click=ImageClick(1.0, 2.0),
        ),
        analysis=AnalysisResult(
            fen=f"{STARTING_PLACEMENT} w KQkq - 0 1",
            top_moves=[],
            engine_depth=20,
            analysis_latency_ms=10.0,
            analysis_status="success",
        ),
        explanation=ExplanationResult(
            move_uci="e2e4",
            move_san="e4",
            explanation_text=None,
            structured_explanation=StructuredExplanation(
                summary="e4 takes the center.",
                what_the_move_does="It claims central space.",
                what_it_threatens="It opens lines for development.",
                why_it_is_best="It keeps the strongest evaluation.",
                why_alternatives_are_worse="The alternatives are slightly slower.",
                alternatives=[],
                tactical_themes=["fork"],
            ),
            provider=None,
            status="success",
        ),
        status="success",
        user_action_required=None,
        warnings=[],
    )
    values = result.__dict__ | overrides
    return CoachingResult(**values)


def test_fen_subcommand_prints_output(capsys) -> None:
    with patch(
        "chesscoach.analysis.coach.ChessCoach.analyze_position",
        return_value=[],
    ):
        with patch(
            "sys.argv",
            ["chess-coach", "fen", STARTING_PLACEMENT, "w", "-", "-", "0", "1"],
        ):
            from chesscoach.cli import main

            main()

    captured = capsys.readouterr()
    assert "Top 0 moves for:" in captured.out


def test_cli_keeps_legacy_fen_mode(capsys) -> None:
    with patch(
        "chesscoach.analysis.coach.ChessCoach.analyze_position",
        return_value=[],
    ):
        with patch(
            "sys.argv", ["chess-coach", STARTING_PLACEMENT, "w", "-", "-", "0", "1"]
        ):
            from chesscoach.cli import main

            main()

    captured = capsys.readouterr()
    assert "Top 0 moves for:" in captured.out


def test_image_subcommand_uses_default_white_and_prints_warning(
    capsys, monkeypatch
) -> None:
    captured_request = {}

    def _run_pipeline(request):
        captured_request["request"] = request
        return make_pipeline_result(
            warnings=[
                PipelineWarning(
                    code="explanation_skipped_unavailable",
                    message="Explanation was skipped because no LLM provider is configured.",
                )
            ]
        )

    monkeypatch.setattr("chesscoach.cli.run_coaching_pipeline", _run_pipeline)

    from chesscoach.cli import main

    main(
        [
            "image",
            "board.jpg",
            "--white-king-start-click-x",
            "12",
            "--white-king-start-click-y",
            "34",
        ]
    )

    captured = capsys.readouterr()
    assert captured_request["request"].side_to_move == "w"
    assert captured_request["request"].explanation_provider is None
    assert "Warning [explanation_skipped_unavailable]" in captured.out
    assert "Summary: e4 takes the center." in captured.out


def test_image_subcommand_json_outputs_machine_readable_payload(
    capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        "chesscoach.cli.run_coaching_pipeline",
        lambda request: make_pipeline_result(),
    )

    from chesscoach.cli import main

    main(
        [
            "image",
            "board.jpg",
            "--white-king-start-click-x",
            "12",
            "--white-king-start-click-y",
            "34",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["position"]["side_to_move"] == "w"
    assert payload["explanation"]["structured_explanation"]["summary"] == (
        "e4 takes the center."
    )


def test_image_subcommand_passes_explanation_provider_and_model(
    capsys, monkeypatch
) -> None:
    captured_request = {}

    def _run_pipeline(request):
        captured_request["request"] = request
        return make_pipeline_result()

    monkeypatch.setattr("chesscoach.cli.run_coaching_pipeline", _run_pipeline)

    from chesscoach.cli import main

    main(
        [
            "image",
            "board.jpg",
            "--white-king-start-click-x",
            "12",
            "--white-king-start-click-y",
            "34",
            "--include-explanation",
            "--explanation-provider",
            "openai",
            "--explanation-model",
            "gpt-4o-mini",
        ]
    )

    _ = capsys.readouterr()
    assert captured_request["request"].include_explanation is True
    assert captured_request["request"].explanation_provider == "openai"
    assert captured_request["request"].explanation_model == "gpt-4o-mini"


def test_image_subcommand_failure_exits(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "chesscoach.cli.run_coaching_pipeline",
        lambda request: make_pipeline_result(
            status="failed",
            position=None,
            analysis=None,
            warnings=[
                PipelineWarning(
                    code="board_detection_low_confidence",
                    message=(
                        "The board could not be detected. Please try to upload "
                        "a clearer image."
                    ),
                )
            ],
        ),
    )

    from chesscoach.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "image",
                "board.jpg",
                "--white-king-start-click-x",
                "12",
                "--white-king-start-click-y",
                "34",
            ]
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "board_detection_low_confidence" in captured.out
