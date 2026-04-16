"""Top-level pipeline: image bytes/path/PIL → FEN piece-placement string."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from chesscoach.vision.board_detector import detect_board, split_into_squares
from chesscoach.vision.fen_builder import build_fen
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.types import PieceLabel, SquareGrid

_default_classifier: PieceClassifier | None = None
LOGGER = logging.getLogger(__name__)


def _get_default_classifier() -> PieceClassifier:
    global _default_classifier
    if _default_classifier is None:
        LOGGER.info("Initializing default piece classifier")
        _default_classifier = PieceClassifier()
    return _default_classifier


def _to_bgr(image: bytes | Path | PILImage.Image) -> np.ndarray:
    """Convert any supported input type to a BGR numpy array."""
    if isinstance(image, (bytes, bytearray)):
        LOGGER.debug("Decoding board image from raw bytes")
        arr = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        return bgr

    if isinstance(image, Path):
        LOGGER.debug(f"Loading board image from path: {image}")
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"Could not read image file: {image}")
        return bgr

    # PIL.Image
    LOGGER.debug("Converting board image from PIL.Image")
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def predict_fen(
    image: bytes | Path | PILImage.Image,
    classifier: PieceClassifier | None = None,
) -> str:
    """Detect the chess position in *image* and return a FEN piece-placement string.

    Only the piece-placement field is returned (the first FEN segment, e.g.
    ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"``).  The caller is
    responsible for appending the remaining FEN fields (active color, castling
    rights, etc.) which cannot be inferred from a single image.

    Args:
        image: The board image as raw bytes, a file :class:`~pathlib.Path`, or
            a :class:`PIL.Image.Image`.
        classifier: A :class:`~chesscoach.vision.piece_classifier.PieceClassifier`
            instance.  Defaults to a stub classifier (returns ``"empty"`` for
            every square) when no checkpoint has been loaded.

    Returns:
        FEN piece-placement string (rank 8 first, rank 1 last).

    Raises:
        :exc:`~chesscoach.vision.board_detector.BoardNotFoundError`: If no
            chessboard can be located in the image.
        :exc:`ValueError`: If the image cannot be decoded.
    """
    if classifier is None:
        classifier = _get_default_classifier()

    LOGGER.info("Starting FEN prediction")
    bgr = _to_bgr(image)
    warped = detect_board(bgr)
    occupancy_squares = split_into_squares(warped, context_scale=1.0)
    piece_squares = split_into_squares(
        warped,
        crop_width_scale=1.5,
        crop_height_scale=2.4,
        center_y_offset_scale=-0.45,
    )

    grid: SquareGrid = []
    for row_idx, (occ_row, piece_row) in enumerate(
        zip(occupancy_squares, piece_squares)
    ):
        rank_labels: list[PieceLabel] = [
            classifier.classify(occ_square, piece_square)
            for occ_square, piece_square in zip(occ_row, piece_row)
        ]
        LOGGER.debug(f"Predicted rank {row_idx} labels: {rank_labels}")
        grid.append(rank_labels)

    fen = build_fen(grid)
    LOGGER.info(f"Finished FEN prediction: {fen}")
    return fen
