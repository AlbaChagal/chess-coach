"""Tests for the image to FEN CLI."""

from __future__ import annotations

from pathlib import Path

from chesscoach import vision_cli


def test_vision_cli_uses_detector_checkpoint(monkeypatch) -> None:
    detector_calls: list[tuple[Path | None, float, int]] = []
    localizer_calls: list[tuple[Path, int]] = []

    class _Detector:
        def __init__(
            self,
            checkpoint: Path | None = None,
            *,
            score_threshold: float = 0.05,
            image_size: int = 640,
        ) -> None:
            detector_calls.append((checkpoint, score_threshold, image_size))

    class _Localizer:
        def __init__(self, checkpoint: Path, *, image_size: int = 512) -> None:
            localizer_calls.append((checkpoint, image_size))

    monkeypatch.setattr(vision_cli, "PieceDetector", _Detector)
    monkeypatch.setattr(vision_cli, "BoardCornerLocalizer", _Localizer)
    monkeypatch.setattr(
        vision_cli,
        "predict_fen",
        lambda image, detector, board_localizer=None: "8/8/8/8/8/8/8/8",
    )
    monkeypatch.setattr(vision_cli, "configure_logging", lambda level: None)

    vision_cli.main(
        [
            "board.png",
            "--detector-checkpoint",
            "models/piece_detector.pt",
            "--board-localizer-checkpoint",
            "models/board_localizer.pt",
            "--board-localizer-image-size",
            "640",
            "--score-threshold",
            "0.07",
            "--image-size",
            "800",
        ]
    )

    assert detector_calls == [(Path("models/piece_detector.pt"), 0.07, 800)]
    assert localizer_calls == [(Path("models/board_localizer.pt"), 640)]
