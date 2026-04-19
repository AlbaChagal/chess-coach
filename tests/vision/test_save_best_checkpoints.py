"""Tests for checkpoint snapshot utility."""

from __future__ import annotations

from pathlib import Path

from scripts.save_best_checkpoints import save_best_checkpoints


def test_save_best_checkpoints_copies_and_updates_notes(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    piece_detector = models_dir / "piece_detector.pt"
    board_localizer = models_dir / "board_localizer.pt"
    piece_detector.write_bytes(b"piece")
    board_localizer.write_bytes(b"board")

    piece_note = models_dir / "piece_detector_test_metrics.md"
    piece_note.write_text(
        "# Piece Detector Baseline Metrics\n\n"
        "Checkpoint evaluated: `models/piece_detector.pt`\n"
    )
    vision_note = models_dir / "vision_system_baseline_metrics.md"
    vision_note.write_text(
        "# Vision System Baseline Metrics\n\n"
        "Board localizer checkpoint evaluated: `models/board_localizer.pt`\n\n"
        "Piece detector checkpoint evaluated: `models/piece_detector.pt`\n"
    )

    saved_piece_detector, saved_board_localizer = save_best_checkpoints(
        piece_detector_checkpoint=piece_detector,
        board_localizer_checkpoint=board_localizer,
        output_dir=models_dir,
        timestamp="20260419_1112",
        piece_detector_metrics_json=None,
        board_localizer_metrics_json=None,
        vision_benchmark_metrics_json=None,
    )

    assert saved_piece_detector.name == "piece_detector_20260419_1112.pt"
    assert saved_board_localizer.name == "board_localizer_20260419_1112.pt"
    assert saved_piece_detector.read_bytes() == b"piece"
    assert saved_board_localizer.read_bytes() == b"board"
    assert "piece_detector_20260419_1112.pt" in piece_note.read_text()
    vision_content = vision_note.read_text()
    assert "board_localizer_20260419_1112.pt" in vision_content
    assert "piece_detector_20260419_1112.pt" in vision_content
