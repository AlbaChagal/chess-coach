"""Regression tests for the training script dataset validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torchvision import transforms

from scripts.train import (
    SquareDataset,
    _compute_classification_metrics,
    _compute_color_metrics,
    _make_class_weights,
    _make_weighted_sampler,
    _validate_dataset_files,
    _validate_dataset_sizes,
)


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


def test_square_dataset_can_merge_train_and_test_splits(tmp_path: Path) -> None:
    csv_path = tmp_path / "squares.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerow({"image_path": "empty/train.jpg", "label": "empty", "split": "train"})
        writer.writerow({"image_path": "empty/val.jpg", "label": "empty", "split": "val"})
        writer.writerow({"image_path": "empty/test.jpg", "label": "empty", "split": "test"})

    dataset = SquareDataset(
        csv_path=csv_path,
        split=("train", "test"),
        label_map={"empty": 0},
        transform=transforms.Compose([]),
        root=tmp_path,
    )

    assert len(dataset) == 2


def test_compute_classification_metrics_reports_macro_scores() -> None:
    metrics = _compute_classification_metrics(
        predictions=[0, 0, 1, 1],
        labels=[0, 1, 1, 1],
        num_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx((1 / 2 + 1) / 2)
    assert metrics["macro_recall"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]


def test_compute_color_metrics_reports_white_and_black_scores() -> None:
    metrics = _compute_color_metrics(
        predictions=[0, 1, 2, 0],
        labels=[0, 0, 2, 3],
        class_names=["wP", "wN", "bP", "bN"],
    )

    assert metrics["white_precision"] == pytest.approx(2 / 3)
    assert metrics["white_recall"] == pytest.approx(1.0)
    assert metrics["black_precision"] == pytest.approx(1.0)
    assert metrics["black_recall"] == pytest.approx(1 / 2)


def test_make_class_weights_is_inverse_frequency() -> None:
    weights = _make_class_weights({0: 10, 1: 5, 2: 2}, num_classes=3)

    assert torch.equal(weights, torch.tensor([0.1, 0.2, 0.5]))


def test_make_weighted_sampler_upsamples_minority_class(tmp_path: Path) -> None:
    csv_path = tmp_path / "squares.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerow({"image_path": "empty/a.jpg", "label": "empty", "split": "train"})
        writer.writerow({"image_path": "empty/b.jpg", "label": "empty", "split": "train"})
        writer.writerow({"image_path": "wP/c.jpg", "label": "wP", "split": "train"})

    dataset = SquareDataset(
        csv_path=csv_path,
        split="train",
        label_map={"empty": 0, "wP": 1},
        transform=transforms.Compose([]),
        root=tmp_path,
    )

    sampler = _make_weighted_sampler(dataset)
    weights = list(sampler.weights.tolist())

    assert weights == pytest.approx([0.5, 0.5, 1.0])
