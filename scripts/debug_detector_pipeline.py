"""Write detector overlay images for qualitative debugging."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_postprocess import (
    count_board_errors,
    find_mismatched_squares,
    rerank_board_candidates,
)
from chesscoach.vision.board_detector import (
    DEFAULT_WARP_MARGIN_RATIO,
)
from chesscoach.vision.piece_assignment import (
    PieceDetection,
    collect_square_candidates_via_homography,
)
from chesscoach.vision.piece_detector import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)
from chesscoach.vision.types import SquareGrid

LOGGER = logging.getLogger(__name__)


def _empty_grid() -> SquareGrid:
    return [["empty" for _ in range(8)] for _ in range(8)]


def _square_to_indices(square: str) -> tuple[int, int]:
    return 8 - int(square[1]), ord(square[0]) - ord("a")


def _draw_grid(image: cv2.typing.MatLike) -> None:
    _ = image


def _draw_board_polygon(
    image: cv2.typing.MatLike,
    board_corners: np.ndarray,
) -> None:
    points = board_corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=(255, 255, 0), thickness=2)


def _draw_detections(
    image: cv2.typing.MatLike,
    detections: list[PieceDetection],
    assignments: SquareGrid,
    board_corners: np.ndarray,
) -> None:
    _ = assignments
    _ = board_corners
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        bottom_center_x = int(round((detection.box[0] + detection.box[2]) / 2))
        bottom_center_y = int(round(detection.box[3]))
        cv2.circle(image, (bottom_center_x, bottom_center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"P:{detection.label} {detection.score:.2f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )


def debug_detector_pipeline(
    manifest_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    split: str,
    limit: int,
    failed_only: bool,
    score_threshold: float,
    image_size: int,
) -> None:
    """Write warped-board overlays for detector debugging."""
    detector = PieceDetector(
        checkpoint,
        score_threshold=score_threshold,
        image_size=image_size,
    )
    root = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] != split:
            continue
        expected = _empty_grid()
        for annotation in record["annotations"]:
            row, col = _square_to_indices(annotation["square"])
            expected[row][col] = annotation["label"]

        image_path = root / record["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing detection image: {image_path}")
        detections = detector.detect(image)
        board_corners = np.array(record["board_corners"], dtype=np.float32)
        square_candidates, _ = collect_square_candidates_via_homography(
            detections,
            board_corners=board_corners,
            margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
        )
        assignments = rerank_board_candidates(square_candidates)
        total_errors, missed, extra, wrong_label = count_board_errors(expected, assignments)
        mismatches = find_mismatched_squares(expected, assignments)
        if failed_only and total_errors == 0:
            continue

        overlay = image.copy()
        _draw_board_polygon(overlay, board_corners)
        _draw_detections(overlay, detections, assignments, board_corners)
        cv2.putText(
            overlay,
            (
                f"errors={total_errors} missed={missed} "
                f"extra={extra} wrong_label={wrong_label}"
            ),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for index, (square, expected_label, predicted_label) in enumerate(mismatches[:8], start=1):
            cv2.putText(
                overlay,
                f"{square}: exp={expected_label} pred={predicted_label}",
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), overlay)
        if mismatches:
            mismatch_path = output_dir / f"{image_path.stem}.txt"
            mismatch_path.write_text(
                "\n".join(
                    f"{square}: expected={expected_label} predicted={predicted_label}"
                    for square, expected_label, predicted_label in mismatches
                )
            )
        written += 1
        if written >= limit:
            break

    LOGGER.info(
        f"Detector debug overlays written count={written} output={output_dir} "
        f"split={split} failed_only={failed_only} "
        f"score_threshold={score_threshold:.2f} "
        f"image_size={image_size}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Write detector debug overlays.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--failed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="failed_only",
    )
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
    debug_detector_pipeline(
        args.manifest,
        args.checkpoint,
        args.output,
        split=args.split,
        limit=args.limit,
        failed_only=args.failed_only,
        score_threshold=args.score_threshold,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
