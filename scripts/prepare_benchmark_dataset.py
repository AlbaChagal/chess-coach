"""Prepare an image-path + FEN benchmark dataset from raw labeled boards."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from chesscoach.logging_utils import add_logging_args, configure_logging

try:
    from scripts.prepare_squares import _read_fen_placement
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.prepare_squares import _read_fen_placement

LOGGER = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = ("*.jpg", "*.png")


def prepare_benchmark_dataset(
    input_dir: Path,
    output_path: Path,
) -> Path:
    """Convert raw labeled board images into benchmark CSV format.

    Args:
        input_dir: Root raw dataset directory containing split subdirectories.
        output_path: CSV path to write with columns ``image_path``, ``fen``, and
            ``split``.

    Returns:
        Path to the written benchmark CSV.
    """
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_dir = input_dir / split
        if not split_dir.exists():
            LOGGER.warning(f"Split directory not found: {split_dir}")
            continue
        image_paths = sorted(
            image_path
            for pattern in _IMAGE_EXTENSIONS
            for image_path in split_dir.glob(pattern)
        )
        LOGGER.info(f"Preparing benchmark split={split} images={len(image_paths)}")
        for image_path in image_paths:
            try:
                fen_placement = _read_fen_placement(image_path)
            except ValueError as exc:
                LOGGER.warning(f"Skipping {image_path.name}: {exc}")
                continue
            if fen_placement is None:
                continue
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "fen": fen_placement,
                    "split": split,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["image_path", "fen", "split"])
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(f"Benchmark dataset written rows={len(rows)} path={output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare a benchmark CSV from raw labeled board images."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chess_boards/benchmark.csv"),
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    prepare_benchmark_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
