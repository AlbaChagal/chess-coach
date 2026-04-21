"""Vision component: detect a chess position from an image and return FEN."""

from pathlib import Path

import numpy as np

from chesscoach.vision.board_detector import BoardNotFoundError
from chesscoach.vision.board_localizer import BoardCornerLocalizer
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.piece_detector import PieceDetector
from chesscoach.vision.predictor import predict_fen

__all__ = [
    "predict_fen",
    "BoardCornerLocalizer",
    "PieceClassifier",
    "PieceDetector",
    "BoardNotFoundError",
    "BoardVision",
]


class BoardVision:
    """High-level interface for board-to-FEN prediction.

    Wraps :func:`predict_fen` with a file-path-based API for convenience.

    Args:
        classifier: Optional detector or legacy classifier instance. Defaults
            to the detector path, which falls back to a stub detector when no
            checkpoint is configured.
        board_localizer: Optional learned board localizer. When omitted, the
            detector path will automatically use ``models/board_localizer.pt``
            if that checkpoint is present, otherwise it falls back to the
            classical board detector.
    """

    def __init__(
        self,
        classifier: PieceClassifier | PieceDetector | None = None,
        board_localizer: BoardCornerLocalizer | None = None,
    ) -> None:
        self._classifier = classifier
        self._board_localizer = board_localizer

    def fen_from_image(
        self,
        image_path: Path,
        white_king_start_click: tuple[float, float] | None = None,
        board_corners: list[tuple[float, float]] | None = None,
    ) -> str:
        """Return the FEN piece-placement string for the board shown in *image_path*.

        Args:
            image_path: Path to a PNG or JPEG image of a chess board.
            white_king_start_click: Optional raw image click used to orient the
                board so the white king start square becomes e1 at the bottom.
            board_corners: Optional user-corrected board corners in raw image
                coordinates ordered as top-left, top-right, bottom-right, bottom-left.

        Returns:
            FEN piece-placement string.

        Raises:
            BoardNotFoundError: If no board is detected in the image.
            ValueError: If the file cannot be read.
        """
        return predict_fen(
            image_path,
            self._classifier,
            self._board_localizer,
            white_king_start_click,
            None if board_corners is None else np.array(board_corners, dtype=np.float32),
        )
