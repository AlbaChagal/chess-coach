"""Fine-tune occupancy and piece ResNet-18 classifiers on labeled square images.

Trains two models sequentially:
1. **Occupancy model** (2 classes: empty / occupied) — 10 epochs.
2. **Piece model** (12 classes: wP..bK, empty squares excluded) — 20 epochs.

Both use Adam with ReduceLROnPlateau and early stopping on validation loss.

Usage::

    uv run python scripts/train.py \\
        --squares data/chess_boards/squares/squares.csv \\
        --output  models/

The CSV produced by ``prepare_squares.py`` is expected with columns:
``image_path, label, split`` (split values: train / val / test).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Literal

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from chesscoach.vision.types import PIECE_LABELS, PieceLabel

Split = Literal["train", "val", "test"]

_PIECE_LABELS_NO_EMPTY: list[PieceLabel] = [lbl for lbl in PIECE_LABELS if lbl != "empty"]
_INPUT_SIZE = 100
_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SquareDataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads labeled 100×100 square images from a CSV manifest."""

    def __init__(
        self,
        csv_path: Path,
        split: Split,
        label_map: dict[str, int],
        transform: transforms.Compose,
        root: Path,
    ) -> None:
        self._root = root
        self._transform = transform
        self._samples: list[tuple[Path, int]] = []

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                label = row["label"]
                if label not in label_map:
                    continue  # skip labels not in this classifier's scope
                self._samples.append((root / row["image_path"], label_map[label]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, class_idx = self._samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read square image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = self._transform(rgb)  # type: ignore[assignment]
        return tensor, class_idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _build_resnet(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _validate_dataset_sizes(
    train_size: int,
    val_size: int,
    *,
    csv_path: Path,
    model_name: str,
) -> None:
    """Fail fast when the manifest does not contain usable training data."""
    if train_size <= 0 and val_size <= 0:
        raise ValueError(
            f"{model_name} training dataset is empty for {csv_path}. "
            "Expected rows with split values 'train' and 'val'. "
            "Rebuild the manifest with scripts/prepare_squares.py."
        )
    if train_size <= 0:
        raise ValueError(
            f"{model_name} training dataset has no 'train' samples in {csv_path}. "
            "Rebuild the manifest with scripts/prepare_squares.py."
        )
    if val_size <= 0:
        raise ValueError(
            f"{model_name} training dataset has no 'val' samples in {csv_path}. "
            "Rebuild the manifest with scripts/prepare_squares.py."
        )


def _validate_dataset_files(
    dataset: SquareDataset,
    *,
    csv_path: Path,
    model_name: str,
    split: Split,
) -> None:
    """Fail fast when the manifest references image files that are missing."""
    missing_paths = [
        image_path for image_path, _ in dataset._samples if not image_path.exists()
    ]
    if not missing_paths:
        return

    sample_paths = ", ".join(str(path) for path in missing_paths[:3])
    raise FileNotFoundError(
        f"{model_name} {split} split in {csv_path} references {len(missing_paths)} "
        f"missing image files. Sample paths: {sample_paths}. "
        "Rebuild the manifest with scripts/prepare_squares.py or use the current "
        "dataset path."
    )


def train_model(
    csv_path: Path,
    root: Path,
    label_map: dict[str, int],
    num_classes: int,
    max_epochs: int,
    output_path: Path,
    device: torch.device,
    model_name: str,
    patience: int = 5,
) -> dict[str, list[float]]:
    """Train a ResNet-18 and save the best checkpoint.

    Args:
        csv_path: Path to the squares CSV manifest.
        root: Root directory of square images (prefix for relative paths in CSV).
        label_map: Maps label string to class index.
        num_classes: Number of output classes.
        max_epochs: Maximum number of training epochs.
        output_path: Where to save the best ``.pt`` state dict.
        device: Torch device to train on.
        model_name: Human-readable model name for error reporting.
        patience: Early-stopping patience (epochs without val loss improvement).

    Returns:
        Dict with ``"train_loss"``, ``"val_loss"``, ``"val_acc"`` history lists.
    """
    train_ds = SquareDataset(csv_path, "train", label_map, _TRAIN_TRANSFORM, root)
    val_ds = SquareDataset(csv_path, "val", label_map, _EVAL_TRANSFORM, root)
    _validate_dataset_sizes(
        len(train_ds),
        len(val_ds),
        csv_path=csv_path,
        model_name=model_name,
    )
    _validate_dataset_files(
        train_ds,
        csv_path=csv_path,
        model_name=model_name,
        split="train",
    )
    _validate_dataset_files(
        val_ds,
        csv_path=csv_path,
        model_name=model_name,
        split="val",
    )

    train_dl = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=_BATCH_SIZE, num_workers=2)

    model = _build_resnet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * len(images)
                correct += (logits.argmax(1) == labels).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"  epoch {epoch:3d}/{max_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(output_path))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs.")
                break

    print(f"  Best checkpoint saved to {output_path}")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train occupancy and piece ResNet-18 classifiers."
    )
    parser.add_argument(
        "--squares",
        type=Path,
        required=True,
        help="Path to the squares.csv manifest produced by prepare_squares.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory to save model checkpoints (default: models/).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to write training history JSON.",
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    root = args.squares.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Occupancy model ---
    print("\n=== Training occupancy model (2 classes: empty / occupied) ===")
    occupancy_label_map = {"empty": 0}
    for label in PIECE_LABELS:
        if label != "empty":
            occupancy_label_map[label] = 1

    occ_history = train_model(
        csv_path=args.squares,
        root=root,
        label_map=occupancy_label_map,
        num_classes=2,
        max_epochs=10,
        output_path=args.output / "occupancy.pt",
        device=device,
        model_name="Occupancy model",
    )

    # --- Piece model ---
    print("\n=== Training piece model (12 classes: wP..bK, empty excluded) ===")
    piece_label_map = {label: idx for idx, label in enumerate(_PIECE_LABELS_NO_EMPTY)}

    piece_history = train_model(
        csv_path=args.squares,
        root=root,
        label_map=piece_label_map,
        num_classes=12,
        max_epochs=20,
        output_path=args.output / "piece.pt",
        device=device,
        model_name="Piece model",
    )

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"occupancy": occ_history, "piece": piece_history}, indent=2)
        )
        print(f"\nTraining history written to {args.log}")

    print("\nTraining complete.")
    print(f"  Occupancy checkpoint : {args.output / 'occupancy.pt'}")
    print(f"  Piece checkpoint     : {args.output / 'piece.pt'}")


if __name__ == "__main__":
    main()
