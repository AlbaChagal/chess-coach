"""Write end-to-end benchmark overlays and error diagnostics for failed boards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import path fallback
    sys.path.append(str(_REPO_ROOT))

from benchmarks.vision.dataset import BoardSample, load_csv, load_json  # noqa: E402
from chesscoach.logging_utils import add_logging_args, configure_logging  # noqa: E402
from chesscoach.vision.board_localizer import (  # noqa: E402
    BoardCornerLocalizer,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
)
from chesscoach.vision.board_postprocess import (  # noqa: E402
    count_board_errors,
    find_mismatched_squares,
    rerank_board_candidates,
)
from chesscoach.vision.fen_builder import build_fen  # noqa: E402
from chesscoach.vision.piece_assignment import (  # noqa: E402
    PieceDetection,
    collect_square_candidates_via_homography,
)
from chesscoach.vision.piece_detector import (  # noqa: E402
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)
from chesscoach.vision.types import PieceLabel, SquareGrid  # noqa: E402

try:
    from scripts.prepare_detection_dataset import select_metadata_corners
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from scripts.prepare_detection_dataset import select_metadata_corners

LOGGER = logging.getLogger(__name__)
_CORNER_ERROR_BUCKETS: list[tuple[str, float | None]] = [
    ("<=10px", 10.0),
    ("<=20px", 20.0),
    ("<=40px", 40.0),
    (">40px", None),
]


def _load_dataset(path: Path) -> list[BoardSample]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_csv(path)


_FEN_CHAR_TO_LABEL: dict[str, PieceLabel] = {
    ".": "empty",
    "P": "wP",
    "N": "wN",
    "B": "wB",
    "R": "wR",
    "Q": "wQ",
    "K": "wK",
    "p": "bP",
    "n": "bN",
    "b": "bB",
    "r": "bR",
    "q": "bQ",
    "k": "bK",
}


def _fen_to_squares(fen_placement: str) -> list[PieceLabel]:
    squares: list[str] = []
    for rank in fen_placement.split("/"):
        for char in rank:
            if char.isdigit():
                squares.extend(["."] * int(char))
            else:
                squares.append(char)
    if len(squares) != 64:
        raise ValueError(
            f"FEN placement does not expand to 64 squares: {fen_placement!r}"
        )
    return [_FEN_CHAR_TO_LABEL[square] for square in squares]


def _fen_to_grid(fen_placement: str) -> SquareGrid:
    squares = _fen_to_squares(fen_placement)
    return [squares[index : index + 8] for index in range(0, 64, 8)]


def _load_expected_corners(image_path: Path) -> np.ndarray | None:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None
    import json

    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        return None
    return select_metadata_corners(payload)


def _mean_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(expected - predicted, axis=1).mean())


def _draw_board_polygon(
    image: cv2.typing.MatLike,
    corners: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> None:
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


def _draw_detections(
    image: cv2.typing.MatLike,
    detections: list[PieceDetection],
) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.score:.2f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )


def _log_corner_error_summary(
    *,
    correct_corner_errors: list[float],
    failed_corner_errors: list[float],
) -> None:
    if correct_corner_errors:
        LOGGER.info(
            f"Correct boards with GT corners: count={len(correct_corner_errors)} "
            f"mean_corner_error_px={float(np.mean(correct_corner_errors)):.2f} "
            f"median_corner_error_px={float(np.median(correct_corner_errors)):.2f}"
        )
    if failed_corner_errors:
        LOGGER.info(
            f"Failed boards with GT corners: count={len(failed_corner_errors)} "
            f"mean_corner_error_px={float(np.mean(failed_corner_errors)):.2f} "
            f"median_corner_error_px={float(np.median(failed_corner_errors)):.2f}"
        )

    combined = [(error, True) for error in correct_corner_errors] + [
        (error, False) for error in failed_corner_errors
    ]
    if not combined:
        return

    LOGGER.info("Board accuracy by corner-error bucket:")
    for label, _ in _CORNER_ERROR_BUCKETS:
        bucket = [
            is_correct
            for error, is_correct in combined
            if _bucket_corner_error(error) == label
        ]
        if not bucket:
            continue
        LOGGER.info(
            f"  {label}: boards={len(bucket)} board_accuracy={sum(bucket) / len(bucket):.3f}"
        )


def _bucket_corner_error(corner_error: float) -> str:
    if corner_error <= 10.0:
        return "<=10px"
    if corner_error <= 20.0:
        return "<=20px"
    if corner_error <= 40.0:
        return "<=40px"
    return ">40px"


def _count_neighbor_square_drifts(
    expected_grid: SquareGrid,
    predicted_grid: SquareGrid,
) -> int:
    """Count same-label pieces that appear to have shifted by one square."""
    drift_count = 0
    for label in sorted(
        {
            cell
            for row in expected_grid + predicted_grid
            for cell in row
            if cell != "empty"
        }
    ):
        expected_positions = [
            (row_idx, col_idx)
            for row_idx, row in enumerate(expected_grid)
            for col_idx, cell in enumerate(row)
            if cell == label and predicted_grid[row_idx][col_idx] != label
        ]
        predicted_positions = [
            (row_idx, col_idx)
            for row_idx, row in enumerate(predicted_grid)
            for col_idx, cell in enumerate(row)
            if cell == label and expected_grid[row_idx][col_idx] != label
        ]
        used_predictions: set[tuple[int, int]] = set()
        for expected_pos in expected_positions:
            match = next(
                (
                    predicted_pos
                    for predicted_pos in predicted_positions
                    if predicted_pos not in used_predictions
                    and _is_neighbor_square(expected_pos, predicted_pos)
                ),
                None,
            )
            if match is None:
                continue
            used_predictions.add(match)
            drift_count += 1
    return drift_count


def _is_neighbor_square(
    square_a: tuple[int, int],
    square_b: tuple[int, int],
) -> bool:
    row_delta = abs(square_a[0] - square_b[0])
    col_delta = abs(square_a[1] - square_b[1])
    return max(row_delta, col_delta) == 1


def _log_error_type_summary_by_corner_bucket(
    *,
    bucket_totals: dict[str, dict[str, float]],
) -> None:
    if not bucket_totals:
        return

    LOGGER.info("Error-type summary by corner-error bucket:")
    for label, _ in _CORNER_ERROR_BUCKETS:
        stats = bucket_totals.get(label)
        if not stats or not stats["boards"]:
            continue
        board_count = stats["boards"]
        LOGGER.info(
            f"  {label}: boards={int(board_count)} "
            f"avg_missed={stats['missed'] / board_count:.3f} "
            f"avg_extra={stats['extra'] / board_count:.3f} "
            f"avg_wrong_label={stats['wrong_label'] / board_count:.3f} "
            f"avg_neighbor_drift={stats['neighbor_drift'] / board_count:.3f} "
            f"avg_total_errors={stats['total_errors'] / board_count:.3f}"
        )


def debug_vision_benchmark(
    dataset_path: Path,
    detector_checkpoint: Path,
    output_dir: Path,
    *,
    split: str | None,
    board_localizer_checkpoint: Path | None,
    failed_only: bool,
    limit: int,
    score_threshold: float,
    image_size: int,
    board_localizer_image_size: int,
) -> None:
    """Write end-to-end benchmark overlays and summarize failure patterns."""
    samples = _load_dataset(dataset_path)
    if split is not None:
        samples = [sample for sample in samples if sample.split == split]

    detector = PieceDetector(
        detector_checkpoint,
        score_threshold=score_threshold,
        image_size=image_size,
    )
    board_localizer = (
        BoardCornerLocalizer(
            board_localizer_checkpoint,
            image_size=board_localizer_image_size,
        )
        if board_localizer_checkpoint is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    correct_corner_errors: list[float] = []
    failed_corner_errors: list[float] = []
    total_correct = 0
    total_failed = 0
    bucket_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "boards": 0.0,
            "missed": 0.0,
            "extra": 0.0,
            "wrong_label": 0.0,
            "neighbor_drift": 0.0,
            "total_errors": 0.0,
        }
    )

    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise FileNotFoundError(f"Missing benchmark image: {sample.image_path}")
        predicted_corners = (
            board_localizer.detect_corners(image)
            if board_localizer is not None
            else None
        )
        if predicted_corners is None:
            continue
        detections = detector.detect(image)
        square_candidates, _ = collect_square_candidates_via_homography(
            detections,
            board_corners=predicted_corners,
        )
        predicted_grid = rerank_board_candidates(square_candidates)
        predicted_fen = build_fen(predicted_grid)
        expected_grid = _fen_to_grid(sample.fen_placement)
        total_errors, missed, extra, wrong_label = count_board_errors(
            expected_grid,
            predicted_grid,
        )
        neighbor_drift = _count_neighbor_square_drifts(expected_grid, predicted_grid)
        mismatches = find_mismatched_squares(expected_grid, predicted_grid)
        is_correct = total_errors == 0
        expected_corners = _load_expected_corners(sample.image_path)
        if expected_corners is not None:
            corner_error = _mean_corner_error(expected_corners, predicted_corners)
            bucket = _bucket_corner_error(corner_error)
            bucket_totals[bucket]["boards"] += 1
            bucket_totals[bucket]["missed"] += missed
            bucket_totals[bucket]["extra"] += extra
            bucket_totals[bucket]["wrong_label"] += wrong_label
            bucket_totals[bucket]["neighbor_drift"] += neighbor_drift
            bucket_totals[bucket]["total_errors"] += total_errors
            if is_correct:
                correct_corner_errors.append(corner_error)
            else:
                failed_corner_errors.append(corner_error)
        else:
            corner_error = None

        if is_correct:
            total_correct += 1
        else:
            total_failed += 1

        if failed_only and is_correct:
            continue

        overlay = image.copy()
        if expected_corners is not None:
            _draw_board_polygon(overlay, expected_corners, color=(255, 255, 0))
        _draw_board_polygon(overlay, predicted_corners, color=(255, 0, 255))
        _draw_detections(overlay, detections)
        header = (
            f"errors={total_errors} missed={missed} extra={extra} "
            f"wrong_label={wrong_label} neighbor_drift={neighbor_drift}"
        )
        cv2.putText(
            overlay,
            header,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if corner_error is not None:
            cv2.putText(
                overlay,
                f"corner_error_px={corner_error:.2f}",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for index, (square, expected_label, predicted_label) in enumerate(
            mismatches[:8],
            start=1,
        ):
            cv2.putText(
                overlay,
                f"{square}: exp={expected_label} pred={predicted_label}",
                (12, 48 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / sample.image_path.name
        cv2.imwrite(str(output_path), overlay)
        (output_dir / f"{sample.image_path.stem}.txt").write_text(
            "\n".join(
                [
                    f"image={sample.image_path}",
                    f"expected_fen={sample.fen_placement}",
                    f"predicted_fen={predicted_fen}",
                    f"errors={total_errors}",
                    f"missed={missed}",
                    f"extra={extra}",
                    f"wrong_label={wrong_label}",
                    f"neighbor_drift={neighbor_drift}",
                    (
                        f"corner_error_px={corner_error:.4f}"
                        if corner_error is not None
                        else "corner_error_px=unavailable"
                    ),
                    "",
                    *[
                        f"{square}: expected={expected_label} predicted={predicted_label}"
                        for square, expected_label, predicted_label in mismatches
                    ],
                ]
            ).strip()
            + "\n"
        )
        written += 1
        if written >= limit:
            break

    LOGGER.info(
        f"Benchmark debug complete split={split or 'all'} "
        f"correct_boards={total_correct} failed_boards={total_failed}"
    )
    _log_corner_error_summary(
        correct_corner_errors=correct_corner_errors,
        failed_corner_errors=failed_corner_errors,
    )
    _log_error_type_summary_by_corner_bucket(bucket_totals=bucket_totals)
    LOGGER.info(
        f"Benchmark debug overlays written count={written} output={output_dir} "
        f"failed_only={failed_only}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Write full-system benchmark debug overlays."
    )
    add_logging_args(parser)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--board-localizer-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--failed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="failed_only",
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
    parser.add_argument(
        "--board-localizer-image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="board_localizer_image_size",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    debug_vision_benchmark(
        args.dataset,
        args.detector_checkpoint,
        args.output,
        split=args.split,
        board_localizer_checkpoint=args.board_localizer_checkpoint,
        failed_only=args.failed_only,
        limit=args.limit,
        score_threshold=args.score_threshold,
        image_size=args.image_size,
        board_localizer_image_size=args.board_localizer_image_size,
    )


if __name__ == "__main__":
    main()
