"""Tests for benchmark dataset preparation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_benchmark_dataset import prepare_benchmark_dataset


def test_prepare_benchmark_dataset_writes_csv_rows(tmp_path: Path) -> None:
    raw_train = tmp_path / "raw" / "train"
    raw_train.mkdir(parents=True)
    image_path = raw_train / "board.jpg"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 200, dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
    )

    output_path = prepare_benchmark_dataset(tmp_path / "raw", tmp_path / "benchmark.csv")

    with output_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["fen"] == "8/8/8/8/8/8/8/8"
    assert rows[0]["split"] == "train"
    assert rows[0]["image_path"] == str(image_path.resolve())
