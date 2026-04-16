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
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from chesscoach import mlops
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.torch_utils import select_device
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

Split = Literal["train", "val", "test"]
DatasetSplit = Split | tuple[Split, ...]

_PIECE_LABELS_NO_EMPTY: list[PieceLabel] = [lbl for lbl in PIECE_LABELS if lbl != "empty"]
_INPUT_SIZE = 100
_BATCH_SIZE = 64
_TRAIN_LOG_EVERY = 100
LOGGER = logging.getLogger(__name__)

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
        split: DatasetSplit,
        label_map: dict[str, int],
        transform: transforms.Compose,
        root: Path,
    ) -> None:
        self._root = root
        self._transform = transform
        self._samples: list[tuple[Path, int]] = []
        splits = (split,) if isinstance(split, str) else split
        self._splits = splits

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] not in splits:
                    continue
                label = row["label"]
                if label not in label_map:
                    continue  # skip labels not in this classifier's scope
                self._samples.append((root / row["image_path"], label_map[label]))

    def __len__(self) -> int:
        return len(self._samples)

    def label_counts(self) -> dict[int, int]:
        """Return the number of samples for each encoded class index."""
        counts = Counter(class_idx for _, class_idx in self._samples)
        return dict(sorted(counts.items()))

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


def _log_dataset_summary(
    dataset: SquareDataset,
    *,
    name: str,
    csv_path: Path,
    root: Path,
) -> None:
    """Log a concise dataset summary."""
    LOGGER.info(
        f"{name} dataset loaded from manifest={csv_path} "
        f"root={root} samples={len(dataset)}"
    )
    LOGGER.debug(f"{name} label distribution: {dataset.label_counts()}")
    LOGGER.debug(f"{name} sample preview: {dataset._samples[:3]}")


def _log_model_summary(
    model: nn.Module,
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> None:
    """Log model architecture details."""
    parameter_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    LOGGER.info(
        f"{model_name} model initialized architecture={model.__class__.__name__} "
        f"num_classes={num_classes} device={device} "
        f"parameters={parameter_count} trainable={trainable_count}"
    )
    LOGGER.debug(f"{model_name} model details: {model}")


def _compute_classification_metrics(
    predictions: list[int],
    labels: list[int],
    *,
    num_classes: int,
) -> dict[str, object]:
    """Compute accuracy, macro precision, and macro recall from predictions."""
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(labels, predictions):
        confusion[truth][pred] += 1

    total = len(labels)
    correct = sum(confusion[idx][idx] for idx in range(num_classes))
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []

    for class_idx in range(num_classes):
        true_positive = confusion[class_idx][class_idx]
        predicted_positive = sum(confusion[row][class_idx] for row in range(num_classes))
        actual_positive = sum(confusion[class_idx][col] for col in range(num_classes))

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)

    macro_precision = sum(per_class_precision) / num_classes
    macro_recall = sum(per_class_recall) / num_classes

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion,
    }


def _compute_color_metrics(
    predictions: list[int],
    labels: list[int],
    *,
    class_names: list[str],
) -> dict[str, float]:
    """Compute white/black precision and recall when class names encode color."""
    color_groups = {
        "white": {idx for idx, name in enumerate(class_names) if name.startswith("w")},
        "black": {idx for idx, name in enumerate(class_names) if name.startswith("b")},
    }
    metrics: dict[str, float] = {}

    for color, class_ids in color_groups.items():
        if not class_ids:
            continue

        true_positive = sum(
            1 for truth, pred in zip(labels, predictions) if truth in class_ids and pred in class_ids
        )
        predicted_positive = sum(1 for pred in predictions if pred in class_ids)
        actual_positive = sum(1 for truth in labels if truth in class_ids)

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0

        metrics[f"{color}_precision"] = precision
        metrics[f"{color}_recall"] = recall

    return metrics


def _make_class_weights(label_counts: dict[int, int], *, num_classes: int) -> torch.Tensor:
    """Build inverse-frequency class weights for cross-entropy loss."""
    weights = []
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 0)
        weights.append(0.0 if count == 0 else 1.0 / count)
    return torch.tensor(weights, dtype=torch.float32)


def _make_weighted_sampler(dataset: SquareDataset) -> WeightedRandomSampler:
    """Create a weighted sampler that upsamples minority classes."""
    label_counts = dataset.label_counts()
    sample_weights = [
        1.0 / label_counts[class_idx]
        for _, class_idx in dataset._samples
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


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
    class_names: list[str],
    learning_rate: float,
    class_weighted_loss: bool = False,
    upsample_minority_classes: bool = False,
    patience: int = 5,
    batch_size: int = _BATCH_SIZE,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
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
        class_names: Human-readable class names indexed by class id.
        learning_rate: Optimizer learning rate.
        class_weighted_loss: Apply inverse-frequency class weighting in the loss.
        upsample_minority_classes: Upsample minority classes in the training loader.
        patience: Early-stopping patience (epochs without val loss improvement).
        batch_size: Mini-batch size for DataLoader.
        on_epoch: Optional callback invoked after each epoch with
            ``(epoch_number, metrics_dict)``.  Use this to log metrics to
            MLflow, W&B, or any other tracker without coupling the training
            loop to a specific framework.

    Returns:
        Dict with ``"train_loss"``, ``"val_loss"``, ``"val_acc"`` history lists.
    """
    train_ds = SquareDataset(
        csv_path,
        ("train", "test"),
        label_map,
        _TRAIN_TRANSFORM,
        root,
    )
    val_ds = SquareDataset(csv_path, "val", label_map, _EVAL_TRANSFORM, root)
    _log_dataset_summary(train_ds, name=f"{model_name} train", csv_path=csv_path, root=root)
    _log_dataset_summary(val_ds, name=f"{model_name} val", csv_path=csv_path, root=root)
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

    train_sampler = _make_weighted_sampler(train_ds) if upsample_minority_classes else None
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
    sampler_name = (
        train_sampler.__class__.__name__ if train_sampler is not None else "shuffle"
    )
    LOGGER.info(
        f"{model_name} dataloaders configured batch_size={batch_size} "
        f"train_batches={len(train_dl)} val_batches={len(val_dl)} "
        f"num_workers=2 sampler={sampler_name}"
    )

    model = _build_resnet(num_classes).to(device)
    _log_model_summary(model, model_name=model_name, num_classes=num_classes, device=device)
    class_weights: torch.Tensor | None = None
    if class_weighted_loss:
        class_weights = _make_class_weights(train_ds.label_counts(), num_classes=num_classes)
        LOGGER.info(
            f"{model_name} using weighted loss "
            f"with class_weights={class_weights.tolist()}"
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    LOGGER.info(
        f"{model_name} optimizer={optimizer.__class__.__name__} "
        f"initial_lr={learning_rate:.6f} "
        f"scheduler={scheduler.__class__.__name__} patience={patience}"
    )
    if upsample_minority_classes:
        LOGGER.info(
            f"{model_name} using weighted random sampling "
            "for minority-class upsampling"
        )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_white_precision": [],
        "val_white_recall": [],
        "val_black_precision": [],
        "val_black_recall": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss = 0.0
        for step, (images, labels) in enumerate(train_dl, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
            if LOGGER.isEnabledFor(logging.DEBUG) and step % _TRAIN_LOG_EVERY == 0:
                LOGGER.debug(
                    f"{model_name} epoch={epoch} train_step={step}/{len(train_dl)} "
                    f"batch_loss={loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )
        train_loss /= len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_predictions: list[int] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for step, (images, labels) in enumerate(val_dl, start=1):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                batch_loss = criterion(logits, labels).item()
                val_loss += batch_loss * len(images)
                predicted = logits.argmax(1)
                val_predictions.extend(predicted.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                if LOGGER.isEnabledFor(logging.DEBUG) and step % _TRAIN_LOG_EVERY == 0:
                    running_metrics = _compute_classification_metrics(
                        val_predictions,
                        val_labels,
                        num_classes=num_classes,
                    )
                    LOGGER.debug(
                        f"{model_name} epoch={epoch} val_step={step}/{len(val_dl)} "
                        f"batch_loss={batch_loss:.4f} "
                        f"running_acc={running_metrics['accuracy']:.4f} "
                        f"running_precision={running_metrics['macro_precision']:.4f} "
                        f"running_recall={running_metrics['macro_recall']:.4f}"
                    )
        val_loss /= len(val_ds)
        val_metrics = _compute_classification_metrics(
            val_predictions,
            val_labels,
            num_classes=num_classes,
        )
        color_metrics = _compute_color_metrics(
            val_predictions,
            val_labels,
            class_names=class_names,
        )
        val_acc = val_metrics["accuracy"]
        val_precision = val_metrics["macro_precision"]
        val_recall = val_metrics["macro_recall"]

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        LOGGER.info(
            f"{model_name} epoch {epoch:3d}/{max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_precision={val_precision:.4f} "
            f"val_recall={val_recall:.4f} elapsed={elapsed:.1f}s "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        if color_metrics:
            LOGGER.info(
                f"{model_name} epoch {epoch:3d}/{max_epochs} "
                f"white_precision={color_metrics.get('white_precision', 0.0):.4f} "
                f"white_recall={color_metrics.get('white_recall', 0.0):.4f} "
                f"black_precision={color_metrics.get('black_precision', 0.0):.4f} "
                f"black_recall={color_metrics.get('black_recall', 0.0):.4f}"
            )
        LOGGER.debug(
            f"{model_name} validation confusion matrix: "
            f"{val_metrics['confusion_matrix']}"
        )
        LOGGER.debug(
            f"{model_name} per-class precision={val_metrics['per_class_precision']} "
            f"recall={val_metrics['per_class_recall']}"
        )
        if color_metrics:
            LOGGER.debug(f"{model_name} color metrics: {color_metrics}")

        epoch_metrics: dict[str, float] = {
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(float(val_acc), 5),  # type: ignore[arg-type]
            "val_precision": round(float(val_precision), 5),  # type: ignore[arg-type]
            "val_recall": round(float(val_recall), 5),  # type: ignore[arg-type]
            "val_white_precision": round(color_metrics.get("white_precision", 0.0), 5),
            "val_white_recall": round(color_metrics.get("white_recall", 0.0), 5),
            "val_black_precision": round(color_metrics.get("black_precision", 0.0), 5),
            "val_black_recall": round(color_metrics.get("black_recall", 0.0), 5),
        }

        history["train_loss"].append(epoch_metrics["train_loss"])
        history["val_loss"].append(epoch_metrics["val_loss"])
        history["val_acc"].append(epoch_metrics["val_acc"])
        history["val_precision"].append(epoch_metrics["val_precision"])
        history["val_recall"].append(epoch_metrics["val_recall"])
        history["val_white_precision"].append(epoch_metrics["val_white_precision"])
        history["val_white_recall"].append(epoch_metrics["val_white_recall"])
        history["val_black_precision"].append(epoch_metrics["val_black_precision"])
        history["val_black_recall"].append(epoch_metrics["val_black_recall"])

        if on_epoch is not None:
            on_epoch(epoch, epoch_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(output_path))
            LOGGER.info(f"{model_name} saved improved checkpoint to {output_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOGGER.info(f"{model_name} early stopping after {epoch} epochs")
                break

    LOGGER.info(f"{model_name} best checkpoint available at {output_path}")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train occupancy and piece ResNet-18 classifiers."
    )
    add_logging_args(parser)
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
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam (default: 3e-4).",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    args.output.mkdir(parents=True, exist_ok=True)
    root = args.squares.parent
    device = select_device()
    LOGGER.info(
        f"Starting training manifest={args.squares} "
        f"output={args.output} device={device}"
    )
    LOGGER.info(
        f"Training hyperparameters batch_size={_BATCH_SIZE} lr={args.lr:.6f}"
    )

    # --- Occupancy model ---
    LOGGER.info("=== Training occupancy model (2 classes: empty / occupied) ===")
    occupancy_label_map = {"empty": 0}
    for label in PIECE_LABELS:
        if label != "empty":
            occupancy_label_map[label] = 1
    LOGGER.debug(f"Occupancy label map: {occupancy_label_map}")
    occupancy_class_names = ["empty", "occupied"]
    occ_checkpoint = args.output / "occupancy.pt"

    occ_params: dict[str, object] = {
        "lr": args.lr,
        "batch_size": _BATCH_SIZE,
        "max_epochs": 10,
        "patience": 5,
        "weighted_loss": True,
        "upsample_minority_classes": False,
        "input_size": _INPUT_SIZE,
        "architecture": "ResNet18",
        "dataset_csv": str(args.squares),
    }
    with mlops.training_run(mlops.EXPERIMENTS["occupancy"], "occupancy-train", occ_params):
        occ_history = train_model(
            csv_path=args.squares,
            root=root,
            label_map=occupancy_label_map,
            num_classes=2,
            max_epochs=10,
            output_path=occ_checkpoint,
            device=device,
            model_name="Occupancy model",
            class_names=occupancy_class_names,
            learning_rate=args.lr,
            class_weighted_loss=True,
            upsample_minority_classes=False,
            on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
        )
        mlops.log_artifact(occ_checkpoint)
        if occ_checkpoint.exists():
            mlops.register_checkpoint(occ_checkpoint, "OccupancyClassifier")

    # --- Piece model ---
    LOGGER.info("=== Training piece model (12 classes: wP..bK, empty excluded) ===")
    piece_label_map = {label: idx for idx, label in enumerate(_PIECE_LABELS_NO_EMPTY)}
    LOGGER.debug(f"Piece label map: {piece_label_map}")
    piece_class_names = list(_PIECE_LABELS_NO_EMPTY)
    piece_checkpoint = args.output / "piece.pt"

    piece_params: dict[str, object] = {
        "lr": args.lr,
        "batch_size": _BATCH_SIZE,
        "max_epochs": 20,
        "patience": 5,
        "weighted_loss": False,
        "upsample_minority_classes": True,
        "input_size": _INPUT_SIZE,
        "architecture": "ResNet18",
        "dataset_csv": str(args.squares),
    }
    with mlops.training_run(mlops.EXPERIMENTS["piece"], "piece-train", piece_params):
        piece_history = train_model(
            csv_path=args.squares,
            root=root,
            label_map=piece_label_map,
            num_classes=12,
            max_epochs=20,
            output_path=piece_checkpoint,
            device=device,
            model_name="Piece model",
            class_names=piece_class_names,  # type: ignore[arg-type]
            learning_rate=args.lr,
            class_weighted_loss=False,
            upsample_minority_classes=True,
            on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
        )
        mlops.log_artifact(piece_checkpoint)
        if piece_checkpoint.exists():
            mlops.register_checkpoint(piece_checkpoint, "PieceClassifier")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"occupancy": occ_history, "piece": piece_history}, indent=2)
        )
        LOGGER.info(f"Training history written to {args.log}")

    LOGGER.info("Training complete")
    LOGGER.info(f"Occupancy checkpoint: {occ_checkpoint}")
    LOGGER.info(f"Piece checkpoint: {piece_checkpoint}")


if __name__ == "__main__":
    main()
