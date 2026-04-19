"""Sweep detector score thresholds against the full vision benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import path fallback
    sys.path.append(str(_REPO_ROOT))

from benchmarks.vision.evaluate import (  # noqa: E402
    _load_dataset,
    run_evaluation,
)
from chesscoach.logging_utils import add_logging_args, configure_logging  # noqa: E402
from chesscoach.vision.board_localizer import (  # noqa: E402
    BoardCornerLocalizer,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
)
from chesscoach.vision.piece_detector import (  # noqa: E402
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def _evaluate_threshold(
    *,
    samples: list[object],
    detector_checkpoint: Path,
    board_localizer_checkpoint: Path | None,
    score_threshold: float,
    image_size: int,
    board_localizer_image_size: int,
) -> dict[str, object]:
    """Evaluate one detector score threshold."""
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
    evaluation = run_evaluation(samples, detector, board_localizer)
    return {
        "score_threshold": score_threshold,
        **evaluation,
    }


def sweep_score_thresholds(
    *,
    dataset_path: Path,
    detector_checkpoint: Path,
    board_localizer_checkpoint: Path | None,
    split: str | None,
    thresholds: list[float],
    image_size: int,
    board_localizer_image_size: int,
    jobs: int = 1,
) -> list[dict[str, object]]:
    """Run the full benchmark for each detector score threshold."""
    samples = _load_dataset(dataset_path)
    if split is not None:
        samples = [sample for sample in samples if sample.split == split]
        LOGGER.info(
            f"Loaded {len(samples)} benchmark samples from {dataset_path} "
            f"after split filter={split}"
        )
    else:
        LOGGER.info(f"Loaded {len(samples)} benchmark samples from {dataset_path}")

    results: list[dict[str, object]] = []
    if jobs <= 1:
        LOGGER.info(
            f"Running threshold sweep sequentially thresholds={thresholds} "
            f"image_size={image_size} board_localizer_image_size={board_localizer_image_size}"
        )
        for threshold in thresholds:
            row = _evaluate_threshold(
                samples=samples,
                detector_checkpoint=detector_checkpoint,
                board_localizer_checkpoint=board_localizer_checkpoint,
                score_threshold=threshold,
                image_size=image_size,
                board_localizer_image_size=board_localizer_image_size,
            )
            results.append(row)
            LOGGER.info(
                f"Threshold {threshold:.3f}: "
                f"board_accuracy={float(row['board_accuracy']):.4f} "
                f"square_accuracy={float(row['square_accuracy']):.4f} "
                f"errors={int(row['n_errors'])}"
            )
        return results

    LOGGER.info(
        f"Running threshold sweep in parallel thresholds={thresholds} jobs={jobs} "
        f"image_size={image_size} board_localizer_image_size={board_localizer_image_size}"
    )
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _evaluate_threshold,
                samples=samples,
                detector_checkpoint=detector_checkpoint,
                board_localizer_checkpoint=board_localizer_checkpoint,
                score_threshold=threshold,
                image_size=image_size,
                board_localizer_image_size=board_localizer_image_size,
            ): threshold
            for threshold in thresholds
        }
        LOGGER.info(f"Submitted {len(futures)} threshold jobs")
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            completed += 1
            threshold = float(row["score_threshold"])
            LOGGER.info(
                f"Completed {completed}/{len(futures)} threshold jobs: "
                f"threshold={threshold:.3f} "
                f"board_accuracy={float(row['board_accuracy']):.4f} "
                f"square_accuracy={float(row['square_accuracy']):.4f} "
                f"errors={int(row['n_errors'])}"
            )

    return sorted(results, key=lambda item: float(item["score_threshold"]))


def _best_result(results: list[dict[str, object]]) -> dict[str, object]:
    """Return the best threshold result by board then square accuracy."""
    return max(
        results,
        key=lambda item: (
            float(item["board_accuracy"]),
            float(item["square_accuracy"]),
            -float(item["score_threshold"]),
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sweep detector score thresholds on the full vision benchmark."
    )
    add_logging_args(parser)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--detector-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--board-localizer-checkpoint",
        type=Path,
        default=None,
        dest="board_localizer_checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=_DEFAULT_THRESHOLDS,
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of thresholds to evaluate in parallel.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    results = sweep_score_thresholds(
        dataset_path=args.dataset,
        detector_checkpoint=args.detector_checkpoint,
        board_localizer_checkpoint=args.board_localizer_checkpoint,
        split=args.split,
        thresholds=args.thresholds,
        image_size=args.image_size,
        board_localizer_image_size=args.board_localizer_image_size,
        jobs=args.jobs,
    )
    best = _best_result(results)
    LOGGER.info(
        f"Best threshold={float(best['score_threshold']):.3f} "
        f"board_accuracy={float(best['board_accuracy']):.4f} "
        f"square_accuracy={float(best['square_accuracy']):.4f}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results, "best": best}, indent=2))
        LOGGER.info(f"Threshold sweep results written to {args.output}")


if __name__ == "__main__":
    main()
