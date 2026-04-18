"""Evaluate the detector using square-assignment metrics on warped boards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_postprocess import (
    count_board_errors,
    rerank_board_candidates,
)
from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    DEFAULT_WARP_MARGIN_RATIO,
)
from chesscoach.vision.piece_assignment import (
    AssignmentStats,
    collect_square_candidates_via_homography,
)
from chesscoach.vision.piece_detector import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)
from chesscoach.vision.types import PIECE_LABELS, PieceLabel, SquareGrid

LOGGER = logging.getLogger(__name__)
_NON_EMPTY_LABELS: list[PieceLabel] = [label for label in PIECE_LABELS if label != "empty"]


def _empty_grid() -> SquareGrid:
    return [["empty" for _ in range(8)] for _ in range(8)]


def _square_to_indices(square: str) -> tuple[int, int]:
    return 8 - int(square[1]), ord(square[0]) - ord("a")


def _init_metric_counters() -> dict[str, Counter[str] | int]:
    return {
        "tp": Counter(),
        "fp": Counter(),
        "fn": Counter(),
        "predicted_pieces": 0,
        "expected_pieces": 0,
        "accepted_detections": 0,
        "same_square_rejections": 0,
        "neighbor_duplicate_rejections": 0,
        "boards_leq_1_error": 0,
        "boards_leq_2_errors": 0,
        "missed_pieces": 0,
        "extra_pieces": 0,
        "wrong_label_pieces": 0,
    }


def _update_classification_counters(
    expected: SquareGrid,
    predicted: SquareGrid,
    counters: dict[str, Counter[str] | int],
) -> None:
    tp = counters["tp"]
    fp = counters["fp"]
    fn = counters["fn"]
    assert isinstance(tp, Counter)
    assert isinstance(fp, Counter)
    assert isinstance(fn, Counter)

    for row in range(8):
        for col in range(8):
            expected_label = expected[row][col]
            predicted_label = predicted[row][col]
            if expected_label != "empty":
                counters["expected_pieces"] += 1  # type: ignore[operator]
            if predicted_label != "empty":
                counters["predicted_pieces"] += 1  # type: ignore[operator]

            if expected_label == predicted_label:
                if expected_label != "empty":
                    tp[expected_label] += 1
                continue

            if predicted_label != "empty":
                fp[predicted_label] += 1
            if expected_label != "empty":
                fn[expected_label] += 1


def _compute_per_class_metrics(
    counters: dict[str, Counter[str] | int],
) -> dict[str, dict[str, float]]:
    tp = counters["tp"]
    fp = counters["fp"]
    fn = counters["fn"]
    assert isinstance(tp, Counter)
    assert isinstance(fp, Counter)
    assert isinstance(fn, Counter)

    metrics: dict[str, dict[str, float]] = {}
    for label in _NON_EMPTY_LABELS:
        true_positive = tp[label]
        false_positive = fp[label]
        false_negative = fn[label]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        support = true_positive + false_negative
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "support": float(support),
        }
    return metrics


def _update_assignment_counters(
    counters: dict[str, Counter[str] | int],
    stats: AssignmentStats,
) -> None:
    counters["accepted_detections"] += stats.accepted_detections  # type: ignore[operator]
    counters["same_square_rejections"] += stats.same_square_rejections  # type: ignore[operator]
    counters["neighbor_duplicate_rejections"] += (  # type: ignore[operator]
        stats.neighbor_duplicate_rejections
    )


def evaluate_detector(
    manifest_path: Path,
    checkpoint: Path,
    *,
    split: str,
    score_threshold: float = 0.35,
    image_size: int = DEFAULT_DETECTOR_IMAGE_SIZE,
) -> dict[str, float | dict[str, dict[str, float]]]:
    """Evaluate detector square assignment on a manifest split."""
    detector = PieceDetector(
        checkpoint,
        score_threshold=score_threshold,
        image_size=image_size,
    )
    root = manifest_path.parent
    correct = 0
    total = 0
    occupied_correct = 0
    occupied_total = 0
    perfect_boards = 0
    board_count = 0
    counters = _init_metric_counters()

    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] != split:
            continue
        board_count += 1
        expected = _empty_grid()
        for annotation in record["annotations"]:
            row, col = _square_to_indices(annotation["square"])
            expected[row][col] = annotation["label"]

        image = cv2.imread(str(root / record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Missing detection image: {root / record['image_path']}")
        square_candidates, assignment_stats = collect_square_candidates_via_homography(
            detector.detect(image),
            board_corners=np.array(record["board_corners"], dtype=np.float32),
            board_size=BOARD_SIZE,
            margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
        )
        predicted = rerank_board_candidates(square_candidates)
        _update_assignment_counters(counters, assignment_stats)
        _update_classification_counters(expected, predicted, counters)
        board_correct = 0
        for row in range(8):
            for col in range(8):
                total += 1
                if expected[row][col] != "empty":
                    occupied_total += 1
                if predicted[row][col] == expected[row][col]:
                    correct += 1
                    board_correct += 1
                    if expected[row][col] != "empty":
                        occupied_correct += 1
        if board_correct == 64:
            perfect_boards += 1
        total_errors, missed, extra, wrong_label = count_board_errors(expected, predicted)
        if total_errors <= 1:
            counters["boards_leq_1_error"] += 1  # type: ignore[operator]
        if total_errors <= 2:
            counters["boards_leq_2_errors"] += 1  # type: ignore[operator]
        counters["missed_pieces"] += missed  # type: ignore[operator]
        counters["extra_pieces"] += extra  # type: ignore[operator]
        counters["wrong_label_pieces"] += wrong_label  # type: ignore[operator]

    predicted_pieces = counters["predicted_pieces"]
    expected_pieces = counters["expected_pieces"]
    accepted_detections = counters["accepted_detections"]
    same_square_rejections = counters["same_square_rejections"]
    neighbor_duplicate_rejections = counters["neighbor_duplicate_rejections"]
    boards_leq_1_error = counters["boards_leq_1_error"]
    boards_leq_2_errors = counters["boards_leq_2_errors"]
    missed_pieces = counters["missed_pieces"]
    extra_pieces = counters["extra_pieces"]
    wrong_label_pieces = counters["wrong_label_pieces"]
    assert isinstance(predicted_pieces, int)
    assert isinstance(expected_pieces, int)
    assert isinstance(accepted_detections, int)
    assert isinstance(same_square_rejections, int)
    assert isinstance(neighbor_duplicate_rejections, int)
    assert isinstance(boards_leq_1_error, int)
    assert isinstance(boards_leq_2_errors, int)
    assert isinstance(missed_pieces, int)
    assert isinstance(extra_pieces, int)
    assert isinstance(wrong_label_pieces, int)
    metrics = {
        "square_accuracy": correct / total if total else 0.0,
        "occupied_square_accuracy": (
            occupied_correct / occupied_total if occupied_total else 0.0
        ),
        "board_accuracy": perfect_boards / board_count if board_count else 0.0,
        "boards_at_most_1_error": boards_leq_1_error / board_count if board_count else 0.0,
        "boards_at_most_2_errors": boards_leq_2_errors / board_count if board_count else 0.0,
        "avg_predicted_pieces_per_board": (
            predicted_pieces / board_count if board_count else 0.0
        ),
        "avg_expected_pieces_per_board": (
            expected_pieces / board_count if board_count else 0.0
        ),
        "avg_assigned_pieces_per_board": (
            accepted_detections / board_count if board_count else 0.0
        ),
        "avg_same_square_rejections_per_board": (
            same_square_rejections / board_count if board_count else 0.0
        ),
        "avg_neighbor_duplicate_rejections_per_board": (
            neighbor_duplicate_rejections / board_count if board_count else 0.0
        ),
        "avg_missed_pieces_per_board": missed_pieces / board_count if board_count else 0.0,
        "avg_extra_pieces_per_board": extra_pieces / board_count if board_count else 0.0,
        "avg_wrong_label_pieces_per_board": (
            wrong_label_pieces / board_count if board_count else 0.0
        ),
        "boards": float(board_count),
        "per_class": _compute_per_class_metrics(counters),
    }
    LOGGER.info(
        f"Detector evaluation split={split} "
        f"square_accuracy={metrics['square_accuracy']:.4f} "
        f"occupied_square_accuracy={metrics['occupied_square_accuracy']:.4f} "
        f"board_accuracy={metrics['board_accuracy']:.4f} boards={board_count} "
        f"boards_at_most_1_error={metrics['boards_at_most_1_error']:.4f} "
        f"boards_at_most_2_errors={metrics['boards_at_most_2_errors']:.4f} "
        f"avg_predicted_pieces={metrics['avg_predicted_pieces_per_board']:.2f} "
        f"avg_expected_pieces={metrics['avg_expected_pieces_per_board']:.2f} "
        f"avg_assigned_pieces={metrics['avg_assigned_pieces_per_board']:.2f} "
        f"avg_same_square_rejections="
        f"{metrics['avg_same_square_rejections_per_board']:.2f} "
        f"avg_neighbor_duplicate_rejections="
        f"{metrics['avg_neighbor_duplicate_rejections_per_board']:.2f} "
        f"avg_missed_pieces={metrics['avg_missed_pieces_per_board']:.2f} "
        f"avg_extra_pieces={metrics['avg_extra_pieces_per_board']:.2f} "
        f"avg_wrong_label_pieces={metrics['avg_wrong_label_pieces_per_board']:.2f} "
        f"score_threshold={score_threshold:.2f} image_size={image_size}"
    )
    for label, label_metrics in metrics["per_class"].items():
        LOGGER.info(
            f"Per-class label={label} precision={label_metrics['precision']:.4f} "
            f"recall={label_metrics['recall']:.4f} support={int(label_metrics['support'])}"
        )
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the piece detector.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        dest="score_threshold",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        dest="image_size",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    metrics = evaluate_detector(
        args.manifest,
        args.checkpoint,
        split=args.split,
        score_threshold=args.score_threshold,
        image_size=args.image_size,
    )
    LOGGER.info(f"Detector evaluation complete metrics={metrics}")


if __name__ == "__main__":
    main()
