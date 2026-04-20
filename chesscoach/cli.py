"""CLI entry points for FEN analysis and image-based coaching."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn

from chesscoach.analysis.coach import ChessCoach
from chesscoach.analysis.engine import ChessEngine
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.pipeline import coaching_result_to_dict, run_coaching_pipeline
from chesscoach.pipeline_models import CoachingRequest, ImageClick

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for position analysis and image-based coaching."""
    normalized_argv = _normalize_legacy_argv(argv or sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(normalized_argv)

    log_stream = sys.stderr if getattr(args, "json", False) else sys.stdout
    configure_logging(args.log_level, stream=log_stream)

    if args.command == "fen":
        _run_fen_command(args)
        return
    if args.command == "image":
        _run_image_command(args)
        return
    if args.command == "serve":
        _run_serve_command(args)
        return
    raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChessCoach command line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fen_parser = subparsers.add_parser("fen", help="Analyze a position from FEN.")
    fen_parser.add_argument("fen", nargs="*", help="FEN string to analyze.")
    add_logging_args(fen_parser)

    image_parser = subparsers.add_parser(
        "image", help="Analyze a chessboard image end-to-end."
    )
    image_parser.add_argument("image", type=Path)
    image_parser.add_argument(
        "--side-to-move",
        choices=["w", "b"],
        default="w",
        dest="side_to_move",
        help="Side to move. Defaults to white.",
    )
    image_parser.add_argument(
        "--white-king-start-click-x",
        type=float,
        required=True,
        dest="white_king_start_click_x",
    )
    image_parser.add_argument(
        "--white-king-start-click-y",
        type=float,
        required=True,
        dest="white_king_start_click_y",
    )
    image_parser.add_argument(
        "--castling-rights",
        default=None,
        dest="castling_rights",
        help="Optional castling rights override (e.g. KQkq or -).",
    )
    image_parser.add_argument(
        "--en-passant",
        default=None,
        dest="en_passant",
        help="Optional en passant square override.",
    )
    image_parser.add_argument(
        "--played-move-uci",
        default=None,
        dest="played_move_uci",
        help="Optional played move in UCI format for coaching mode.",
    )
    image_parser.add_argument(
        "--include-explanation",
        action="store_true",
        help="Attempt to explain the best engine move.",
    )
    image_parser.add_argument(
        "--explanation-provider",
        choices=["anthropic", "openai"],
        default=None,
        dest="explanation_provider",
        help="Optional explanation provider override.",
    )
    image_parser.add_argument(
        "--explanation-model",
        default=None,
        dest="explanation_model",
        help="Optional explanation model override.",
    )
    image_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout.",
    )
    add_logging_args(image_parser)

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP API.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    add_logging_args(serve_parser)
    return parser


def _run_fen_command(args: argparse.Namespace) -> None:
    if args.fen:
        fen = " ".join(args.fen)
    else:
        fen = input("Enter FEN: ").strip()

    engine = ChessEngine()
    coach = ChessCoach(engine)

    try:
        moves = coach.analyze_position(fen)
    except ValueError as exc:
        LOGGER.error(f"Invalid analysis request: {exc}")
        raise SystemExit(1) from exc
    finally:
        engine.close()

    LOGGER.info(f"{coach.format_suggestions(fen, moves)}")


def _run_image_command(args: argparse.Namespace) -> None:
    request = CoachingRequest(
        image=args.image,
        side_to_move=args.side_to_move,
        white_king_start_click=ImageClick(
            x=args.white_king_start_click_x,
            y=args.white_king_start_click_y,
        ),
        castling_rights=args.castling_rights,
        en_passant=args.en_passant,
        played_move_uci=args.played_move_uci,
        include_explanation=args.include_explanation,
        explanation_provider=args.explanation_provider,
        explanation_model=args.explanation_model,
    )
    result = run_coaching_pipeline(request)

    if args.json:
        print(json.dumps(coaching_result_to_dict(result), indent=2))
        return

    _print_human_readable_result(result)
    if result.status == "failed":
        raise SystemExit(1)


def _run_serve_command(args: argparse.Namespace) -> None:
    """Run the ChessCoach HTTP server for local/mobile testing."""
    uvicorn.run(
        "chesscoach.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


def _print_human_readable_result(result) -> None:
    print(f"Status: {result.status}")
    if result.user_action_required is not None:
        print(f"User action required: {result.user_action_required}")

    if result.vision.fen_placement is not None:
        print(f"Piece placement: {result.vision.fen_placement}")
    if result.vision.vision_confidence is not None:
        print(f"Vision confidence: {result.vision.vision_confidence:.1f}")

    if result.position is not None:
        print(f"FEN: {result.position.fen}")

    if result.analysis is not None:
        print("Top moves:")
        for index, move in enumerate(result.analysis.top_moves, start=1):
            line = " ".join(move.continuation) if move.continuation else "-"
            print(f"{index}. {move.move_san} [{move.score_display()}] line: {line}")

    if result.explanation is not None and result.explanation.played_move_result:
        played = result.explanation.played_move_result
        comparison = result.explanation.comparison
        print("Played move coaching:")
        print(f"Played move: {played.move_san}")
        print(f"Quality: {played.quality_label} {played.quality_emoji}".strip())
        print(f"Centipawn loss: {played.cp_loss}")
        if comparison is not None:
            print(
                f"Best move: {comparison.best_move_san} "
                f"[{comparison.best_move_score_display}]"
            )
            print(f"Why best was better: {comparison.why_best_move_is_better}")

    if result.explanation is not None and result.explanation.structured_explanation:
        structured = result.explanation.structured_explanation
        print("Explanation:")
        if result.explanation.played_move_result is None:
            print(f"Summary: {structured.summary}")
            print(f"What it does: {structured.what_the_move_does}")
            print(f"Threat: {structured.what_it_threatens}")
            print(f"Why best: {structured.why_it_is_best}")
            print(
                f"Why alternatives are worse: {structured.why_alternatives_are_worse}"
            )
            if structured.tactical_themes:
                print(f"Tactical themes: {', '.join(structured.tactical_themes)}")
            if structured.alternatives:
                print("Alternatives:")
                for alternative in structured.alternatives:
                    print(
                        f"- {alternative.move_san} [{alternative.score_display}]: "
                        f"{alternative.reason}"
                    )
        else:
            print(f"Summary: {structured.summary}")
            print(f"Move intent: {structured.what_the_move_tried_to_do}")
            print(f"What was missed: {structured.what_was_missed}")
            print(f"What changed: {structured.what_changed_after_move}")
            print(f"Why best was better: {structured.why_best_move_was_better}")
            print(f"Lesson: {structured.practical_lesson}")
            if structured.tactical_themes:
                print(f"Tactical themes: {', '.join(structured.tactical_themes)}")
            if structured.alternatives:
                print("Alternatives:")
                for alternative in structured.alternatives:
                    print(
                        f"- {alternative.move_san} [{alternative.score_display}]: "
                        f"{alternative.reason}"
                    )

    if result.explanation is not None and result.explanation.explanation_text:
        print("Narrative:")
        print(result.explanation.explanation_text)

    for warning in result.warnings:
        print(f"Warning [{warning.code}]: {warning.message}")


def _normalize_legacy_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["fen"]
    if argv[0] not in {"fen", "image", "serve"}:
        return ["fen", *argv]
    return argv


if __name__ == "__main__":
    main()
