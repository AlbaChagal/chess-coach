"""Convert raw chesscog board images + FEN labels into labeled 100×100 square images.

For each board image the script:
1. Runs :func:`~chesscoach.vision.board_detector.detect_board` to produce a
   warped 512×512 top-down view.
2. Splits it into 64 squares.
3. Labels each square from the FEN string.
4. Saves the 100×100 square image under ``output/{label}/``.
5. Records the sample in a CSV manifest (``output/squares.csv``).

Boards that fail detection are skipped and logged to stderr.

Usage::

    uv run python scripts/prepare_squares.py \\
        --input  data/chesscog/raw \\
        --output data/chesscog/squares

Expected input layout (chesscog convention)::

    data/chesscog/raw/
        train/
            <id>.jpg   (board image)
            <id>.fen   (FEN string, first line used)
        val/
            ...
        test/
            ...

Output layout::

    data/chesscog/squares/
        wP/  wN/  wB/  wR/  wQ/  wK/
        bP/  bN/  bB/  bR/  bQ/  bK/
        empty/
        squares.csv     # columns: image_path, label, split
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import BoardNotFoundError, detect_board, split_into_squares
from chesscoach.vision.fen_builder import _LABEL_TO_FEN  # reuse the label mapping
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

_SQUARE_SIZE = 100
LOGGER = logging.getLogger(__name__)

# Invert the label→FEN char map to get FEN char→label
_FEN_TO_LABEL: dict[str, PieceLabel] = {
    v: k for k, v in _LABEL_TO_FEN.items() if v != ""
}
_FEN_TO_LABEL["."] = "empty"


def _fen_to_grid(fen_placement: str) -> list[list[PieceLabel]]:
    """Expand a FEN piece-placement string into an 8×8 label grid."""
    grid: list[list[PieceLabel]] = []
    for rank_str in fen_placement.split("/"):
        row: list[PieceLabel] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["empty"] * int(ch))  # type: ignore[arg-type]
            else:
                label = _FEN_TO_LABEL.get(ch)
                if label is None:
                    raise ValueError(f"Unknown FEN character: {ch!r}")
                row.append(label)
        if len(row) != 8:
            raise ValueError(f"Rank does not have 8 squares: {rank_str!r}")
        grid.append(row)
    if len(grid) != 8:
        raise ValueError(f"FEN does not have 8 ranks: {fen_placement!r}")
    return grid


def _read_fen_placement(image_path: Path) -> str | None:
    """Read the piece-placement FEN for a board image sidecar."""
    fen_path = image_path.with_suffix(".fen")
    if fen_path.exists():
        return fen_path.read_text().strip().split()[0]

    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None

    payload = json.loads(json_path.read_text())
    fen = payload.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError(f"Missing 'fen' string in {json_path}")
    return fen.strip().split()[0]


def _log_split_summary(split: str, counters: Counter[str]) -> None:
    """Log a compact per-split processing summary."""
    total_images = counters["total_images"]
    success_boards = counters["success_boards"]
    saved_squares = counters["saved_squares"]
    LOGGER.info(
        f"Split={split} summary total_images={total_images} "
        f"labeled_images={counters['labeled_images']} success_boards={success_boards} "
        f"saved_squares={saved_squares}"
    )
    LOGGER.info(
        f"Split={split} failures missing_labels={counters['missing_labels']} "
        f"unreadable={counters['unreadable_images']} "
        f"board_not_found={counters['board_not_found']} bad_fen={counters['bad_fen']} "
        f"bad_label={counters['bad_label_files']}"
    )
    if total_images:
        LOGGER.info(
            f"Split={split} retention labeled={100 * counters['labeled_images'] / total_images:.1f}% "
            f"detected={100 * success_boards / total_images:.1f}% "
            f"saved={100 * success_boards / total_images:.1f}%"
        )


def prepare_squares(input_dir: Path, output_dir: Path) -> None:
    """Convert raw board images to labeled square images.

    Args:
        input_dir: Root of the raw chesscog dataset (contains train/val/test).
        output_dir: Root of the output square image dataset.
    """
    LOGGER.info(f"Preparing square dataset input={input_dir} output={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in PIECE_LABELS:
        (output_dir / label).mkdir(exist_ok=True)

    csv_path = output_dir / "squares.csv"
    splits = ["train", "val", "test"]

    n_saved = 0
    n_skipped = 0
    split_counters: dict[str, Counter[str]] = {split: Counter() for split in splits}

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()

        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                LOGGER.warning(f"Split directory not found: {split_dir}")
                continue

            image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
            LOGGER.info(f"Processing split={split} images={len(image_paths)}")
            split_counters[split]["total_images"] = len(image_paths)
            for img_path in tqdm(image_paths, desc=split):
                try:
                    fen_placement = _read_fen_placement(img_path)
                except ValueError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to bad label file: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["bad_label_files"] += 1
                    continue

                if fen_placement is None:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to missing .fen/.json label file"
                    )
                    n_skipped += 1
                    split_counters[split]["missing_labels"] += 1
                    continue
                split_counters[split]["labeled_images"] += 1

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    LOGGER.warning(f"Skipping unreadable image: {img_path}")
                    n_skipped += 1
                    split_counters[split]["unreadable_images"] += 1
                    continue

                try:
                    warped = detect_board(bgr)
                except BoardNotFoundError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} because board was not found: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["board_not_found"] += 1
                    continue

                try:
                    grid = _fen_to_grid(fen_placement)
                except ValueError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to invalid FEN: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["bad_fen"] += 1
                    continue

                squares = split_into_squares(warped)
                for row_idx, (sq_row, label_row) in enumerate(zip(squares, grid)):
                    for col_idx, (sq, label) in enumerate(zip(sq_row, label_row)):
                        resized = cv2.resize(sq, (_SQUARE_SIZE, _SQUARE_SIZE))
                        sq_filename = f"{img_path.stem}_r{row_idx}c{col_idx}.jpg"
                        sq_path = output_dir / label / sq_filename
                        cv2.imwrite(str(sq_path), resized)
                        writer.writerow(
                            {
                                "image_path": str(sq_path.relative_to(output_dir)),
                                "label": label,
                                "split": split,
                            }
                        )
                        n_saved += 1
                        split_counters[split]["saved_squares"] += 1
                split_counters[split]["success_boards"] += 1
                LOGGER.debug(
                    f"Saved 64 labeled squares for {img_path.name} split={split}"
                )

    for split in splits:
        _log_split_summary(split, split_counters[split])
    LOGGER.info(f"Square preparation complete saved={n_saved} skipped={n_skipped}")
    LOGGER.info(f"Manifest written to {csv_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert chesscog board images into labeled 100×100 square images."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root of the raw chesscog dataset (contains train/val/test sub-dirs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chesscog/squares"),
        help="Output directory for square images and manifest CSV.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    prepare_squares(args.input, args.output)


if __name__ == "__main__":
    main()
