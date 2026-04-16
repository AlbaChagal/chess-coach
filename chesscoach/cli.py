from __future__ import annotations

import argparse
import logging

from chesscoach.analysis.coach import ChessCoach
from chesscoach.analysis.engine import ChessEngine
from chesscoach.logging_utils import add_logging_args, configure_logging

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for move suggestions."""
    parser = argparse.ArgumentParser(description="Analyze a chess position from FEN.")
    parser.add_argument("fen", nargs="*", help="FEN string to analyze.")
    add_logging_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

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

    LOGGER.info(f"{coach.format_suggestions(fen, moves)}")


if __name__ == "__main__":
    main()
