"""Snapshot best checkpoints and refresh baseline metric notes."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_timestamp() -> str:
    """Return a timestamp suitable for checkpoint snapshot names."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def _copy_checkpoint(checkpoint: Path, output_dir: Path, timestamp: str) -> Path:
    """Copy a checkpoint to a timestamped snapshot path."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{checkpoint.stem}_{timestamp}{checkpoint.suffix}"
    shutil.copy2(checkpoint, destination)
    return destination


def _load_metrics(path: Path | None) -> dict[str, Any] | None:
    """Load optional metrics JSON."""
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metrics JSON: {path}")
    return payload


def _replace_checkpoint_references(
    content: str,
    replacements: dict[str, str],
) -> str:
    """Replace checkpoint paths in markdown content."""
    updated = content
    for source, destination in replacements.items():
        updated = updated.replace(source, destination)
    return updated


def _note_replacements(
    original_checkpoint: Path,
    saved_checkpoint: Path,
) -> dict[str, str]:
    """Build replacements for both absolute and repo-relative checkpoint paths."""
    return {
        str(original_checkpoint): str(saved_checkpoint),
        f"models/{original_checkpoint.name}": f"models/{saved_checkpoint.name}",
    }


def _render_piece_detector_metrics(
    checkpoint_path: Path,
    metrics: dict[str, Any],
) -> str:
    """Render the piece-detector metrics note from JSON metrics."""
    aggregate_keys = [
        "square_accuracy",
        "occupied_square_accuracy",
        "board_accuracy",
        "boards_at_most_1_error",
        "boards_at_most_2_errors",
        "avg_predicted_pieces_per_board",
        "avg_expected_pieces_per_board",
        "avg_assigned_pieces_per_board",
        "avg_same_square_rejections_per_board",
        "avg_neighbor_duplicate_rejections_per_board",
        "avg_missed_pieces_per_board",
        "avg_extra_pieces_per_board",
        "avg_wrong_label_pieces_per_board",
    ]
    lines = [
        "# Piece Detector Baseline Metrics",
        "",
        f"Checkpoint evaluated: `{checkpoint_path}`",
        "",
        "Evaluation command:",
        "",
        "```bash",
        "uv run python scripts/evaluate_detector.py \\",
        "  --manifest data/chess_boards/detection/manifest.jsonl \\",
        f"  --checkpoint {checkpoint_path} \\",
        "  --split test \\",
        "  --score-threshold 0.05 \\",
        "  --image-size 800 \\",
        "  --log-level INFO",
        "```",
        "",
        "Recorded on `test` split:",
        "",
    ]
    for key in aggregate_keys:
        if key in metrics:
            lines.append(f"- `{key}={metrics[key]}`")

    per_class = metrics.get("per_class", {})
    if isinstance(per_class, dict):
        lines.extend(["", "Per-class metrics:", ""])
        for label, values in per_class.items():
            if not isinstance(values, dict):
                continue
            lines.append(
                f"- `{label}`: `precision={values.get('precision')}`"
                f", `recall={values.get('recall')}`"
                f", `support={values.get('support')}`"
            )
    return "\n".join(lines) + "\n"


def _render_vision_system_metrics(
    board_localizer_path: Path,
    piece_detector_path: Path,
    board_localizer_metrics: dict[str, Any] | None,
    benchmark_metrics: dict[str, Any] | None,
) -> str:
    """Render the full-system baseline note from JSON metrics."""
    lines = [
        "# Vision System Baseline Metrics",
        "",
        f"Board localizer checkpoint evaluated: `{board_localizer_path}`",
        "",
        f"Piece detector checkpoint evaluated: `{piece_detector_path}`",
        "",
        "## Board Localizer Test Metrics",
        "",
        "Evaluation command:",
        "",
        "```bash",
        "uv run python scripts/evaluate_board_localizer.py \\",
        "  --manifest data/chess_boards/board_localizer/manifest.jsonl \\",
        f"  --checkpoint {board_localizer_path} \\",
        "  --split test \\",
        "  --image-size 640 \\",
        "  --log-level INFO",
        "```",
        "",
        "Recorded on `test` split:",
        "",
    ]
    if board_localizer_metrics is not None:
        for key in [
            "mean_corner_error_px",
            "median_corner_error_px",
            "max_corner_error_px",
            "boards_leq_20px_mean_error",
        ]:
            if key in board_localizer_metrics:
                lines.append(f"- `{key}={board_localizer_metrics[key]}`")

    lines.extend(
        [
            "",
            "## Full-System Benchmark Metrics",
            "",
            "Evaluation command:",
            "",
            "```bash",
            "uv run python -m benchmarks.vision.evaluate \\",
            "  --dataset data/chess_boards/benchmark.csv \\",
            f"  --detector-checkpoint {piece_detector_path} \\",
            f"  --board-localizer-checkpoint {board_localizer_path} \\",
            "  --board-localizer-image-size 640 \\",
            "  --split test \\",
            "  --log-level INFO",
            "```",
            "",
            "Recorded on `test` split:",
            "",
        ]
    )
    if benchmark_metrics is not None:
        for key in ["board_accuracy", "square_accuracy"]:
            if key in benchmark_metrics:
                lines.append(f"- `{key}={benchmark_metrics[key]}`")
        per_piece = benchmark_metrics.get("per_piece_accuracy", {})
        if isinstance(per_piece, dict):
            lines.extend(["", "Per-piece accuracy:", ""])
            for label, value in per_piece.items():
                lines.append(f"- `{label}={value}`")
    return "\n".join(lines) + "\n"


def _update_piece_detector_note(
    note_path: Path,
    original_checkpoint: Path,
    saved_checkpoint: Path,
    metrics: dict[str, Any] | None,
) -> None:
    """Update the piece-detector markdown note."""
    if metrics is not None:
        note_path.write_text(_render_piece_detector_metrics(saved_checkpoint, metrics))
        return
    content = note_path.read_text()
    updated = _replace_checkpoint_references(
        content,
        _note_replacements(original_checkpoint, saved_checkpoint),
    )
    note_path.write_text(updated)


def _update_vision_system_note(
    note_path: Path,
    original_board_localizer: Path,
    saved_board_localizer: Path,
    original_piece_detector: Path,
    saved_piece_detector: Path,
    board_localizer_metrics: dict[str, Any] | None,
    benchmark_metrics: dict[str, Any] | None,
) -> None:
    """Update the full-system markdown note."""
    if board_localizer_metrics is not None or benchmark_metrics is not None:
        note_path.write_text(
            _render_vision_system_metrics(
                saved_board_localizer,
                saved_piece_detector,
                board_localizer_metrics,
                benchmark_metrics,
            )
        )
        return
    content = note_path.read_text()
    updated = _replace_checkpoint_references(
        content,
        {
            **_note_replacements(original_board_localizer, saved_board_localizer),
            **_note_replacements(original_piece_detector, saved_piece_detector),
        },
    )
    note_path.write_text(updated)


def save_best_checkpoints(
    *,
    piece_detector_checkpoint: Path,
    board_localizer_checkpoint: Path,
    output_dir: Path,
    timestamp: str,
    piece_detector_metrics_json: Path | None,
    board_localizer_metrics_json: Path | None,
    vision_benchmark_metrics_json: Path | None,
) -> tuple[Path, Path]:
    """Copy best checkpoints and update the baseline metric notes."""
    saved_piece_detector = _copy_checkpoint(
        piece_detector_checkpoint,
        output_dir,
        timestamp,
    )
    saved_board_localizer = _copy_checkpoint(
        board_localizer_checkpoint,
        output_dir,
        timestamp,
    )

    piece_detector_metrics = _load_metrics(piece_detector_metrics_json)
    board_localizer_metrics = _load_metrics(board_localizer_metrics_json)
    vision_benchmark_metrics = _load_metrics(vision_benchmark_metrics_json)

    _update_piece_detector_note(
        output_dir / "piece_detector_test_metrics.md",
        piece_detector_checkpoint,
        saved_piece_detector,
        piece_detector_metrics,
    )
    _update_vision_system_note(
        output_dir / "vision_system_baseline_metrics.md",
        board_localizer_checkpoint,
        saved_board_localizer,
        piece_detector_checkpoint,
        saved_piece_detector,
        board_localizer_metrics,
        vision_benchmark_metrics,
    )
    return saved_piece_detector, saved_board_localizer


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Copy best checkpoints and refresh baseline notes."
    )
    parser.add_argument(
        "--piece-detector-checkpoint",
        type=Path,
        default=Path("models/piece_detector.pt"),
        dest="piece_detector_checkpoint",
    )
    parser.add_argument(
        "--board-localizer-checkpoint",
        type=Path,
        default=Path("models/board_localizer.pt"),
        dest="board_localizer_checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        dest="output_dir",
    )
    parser.add_argument("--timestamp", type=str, default=_default_timestamp())
    parser.add_argument(
        "--piece-detector-metrics-json",
        type=Path,
        default=None,
        dest="piece_detector_metrics_json",
    )
    parser.add_argument(
        "--board-localizer-metrics-json",
        type=Path,
        default=None,
        dest="board_localizer_metrics_json",
    )
    parser.add_argument(
        "--vision-benchmark-metrics-json",
        type=Path,
        default=None,
        dest="vision_benchmark_metrics_json",
    )
    args = parser.parse_args(argv)
    saved_piece_detector, saved_board_localizer = save_best_checkpoints(
        piece_detector_checkpoint=args.piece_detector_checkpoint,
        board_localizer_checkpoint=args.board_localizer_checkpoint,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        piece_detector_metrics_json=args.piece_detector_metrics_json,
        board_localizer_metrics_json=args.board_localizer_metrics_json,
        vision_benchmark_metrics_json=args.vision_benchmark_metrics_json,
    )
    print(saved_piece_detector)
    print(saved_board_localizer)


if __name__ == "__main__":
    main()
