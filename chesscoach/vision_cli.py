"""CLI entry point for image to FEN prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision import BoardNotFoundError, predict_fen
from chesscoach.vision.board_localizer import (
    BoardCornerLocalizer,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
)
from chesscoach.vision.piece_detector import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Predict the piece-placement FEN for a board image."""
    parser = argparse.ArgumentParser(description="Predict FEN from a board image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument("--board-localizer-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--board-localizer-image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="board_localizer_image_size",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        dest="score_threshold",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        dest="image_size",
    )
    add_logging_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    detector = (
        PieceDetector(
            args.detector_checkpoint,
            score_threshold=args.score_threshold,
            image_size=args.image_size,
        )
        if args.detector_checkpoint is not None
        else PieceDetector()
    )
    board_localizer = (
        BoardCornerLocalizer(
            args.board_localizer_checkpoint,
            image_size=args.board_localizer_image_size,
        )
        if args.board_localizer_checkpoint is not None
        else None
    )
    try:
        fen = predict_fen(args.image, detector, board_localizer)
    except (BoardNotFoundError, ValueError) as exc:
        LOGGER.error(f"Could not predict FEN for {args.image}: {exc}")
        raise SystemExit(1) from exc

    LOGGER.info(f"Predicted FEN: {fen}")


if __name__ == "__main__":
    main()
