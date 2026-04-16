"""Regression tests for the training script dataset validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from torchvision import transforms

from scripts.train import SquareDataset, _validate_dataset_files, _validate_dataset_sizes


def test_validate_dataset_sizes_rejects_empty_manifest() -> None:
    with pytest.raises(ValueError, match="dataset is empty"):
        _validate_dataset_sizes(
            0,
            0,
            csv_path=Path("data/chesscog/squares/squares.csv"),
            model_name="Occupancy model",
        )


def test_validate_dataset_sizes_requires_train_samples() -> None:
    with pytest.raises(ValueError, match="no 'train' samples"):
        _validate_dataset_sizes(
            0,
            4,
            csv_path=Path("data/chesscog/squares/squares.csv"),
            model_name="Occupancy model",
        )


def test_validate_dataset_sizes_requires_val_samples() -> None:
    with pytest.raises(ValueError, match="no 'val' samples"):
        _validate_dataset_sizes(
            4,
            0,
            csv_path=Path("data/chesscog/squares/squares.csv"),
            model_name="Occupancy model",
        )


def test_validate_dataset_files_rejects_missing_images(tmp_path: Path) -> None:
    csv_path = tmp_path / "squares.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "empty/missing.jpg",
                "label": "empty",
                "split": "train",
            }
        )

    dataset = SquareDataset(
        csv_path=csv_path,
        split="train",
        label_map={"empty": 0},
        transform=transforms.Compose([]),
        root=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="missing image files"):
        _validate_dataset_files(
            dataset,
            csv_path=csv_path,
            model_name="Occupancy model",
            split="train",
        )
