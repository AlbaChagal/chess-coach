"""Vision component: detect a chess position from an image and return FEN."""

from pathlib import Path

from chesscoach.vision.board_detector import BoardNotFoundError
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.piece_detector import PieceDetector
from chesscoach.vision.predictor import predict_fen

__all__ = [
    "predict_fen",
    "PieceClassifier",
    "PieceDetector",
    "BoardNotFoundError",
    "BoardVision",
]


class BoardVision:
    """High-level interface for board-to-FEN prediction.

    Wraps :func:`predict_fen` with a file-path-based API for convenience.

    Args:
        classifier: Optional :class:`PieceClassifier` instance.  Defaults to
            the stub classifier (no checkpoint required).
    """

    def __init__(
        self,
        classifier: PieceClassifier | PieceDetector | None = None,
    ) -> None:
        self._classifier = classifier

    def fen_from_image(self, image_path: Path) -> str:
        """Return the FEN piece-placement string for the board shown in *image_path*.

        Args:
            image_path: Path to a PNG or JPEG image of a chess board.

        Returns:
            FEN piece-placement string.

        Raises:
            BoardNotFoundError: If no board is detected in the image.
            ValueError: If the file cannot be read.
        """
        return predict_fen(image_path, self._classifier)
