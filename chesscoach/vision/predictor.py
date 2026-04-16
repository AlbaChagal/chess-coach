"""Top-level pipeline: image bytes/path/PIL → FEN piece-placement string."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from chesscoach.vision.board_detector import detect_board, split_into_squares
from chesscoach.vision.fen_builder import build_fen
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.types import PieceLabel, SquareGrid

_default_classifier: PieceClassifier | None = None


def _get_default_classifier() -> PieceClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = PieceClassifier()
    return _default_classifier


def _to_bgr(image: bytes | Path | PILImage.Image) -> np.ndarray:
    """Convert any supported input type to a BGR numpy array."""
    if isinstance(image, (bytes, bytearray)):
        arr = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        return bgr

    if isinstance(image, Path):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"Could not read image file: {image}")
        return bgr

    # PIL.Image
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

    bgr = _to_bgr(image)
    warped = detect_board(bgr)
    squares = split_into_squares(warped)

    grid: SquareGrid = []
    for row in squares:
        rank_labels: list[PieceLabel] = [classifier.classify(sq) for sq in row]
        grid.append(rank_labels)

    return build_fen(grid)
