"""Train a warped-board piece detector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.mlops import EXPERIMENTS, log_epoch_metrics, log_artifact, training_run
from chesscoach.vision.detection_dataset import (
    DetectionDataset,
    detection_collate_fn,
)
from chesscoach.vision.piece_detector import (
    DETECTOR_ARCHITECTURE,
    DEFAULT_DETECTOR_IMAGE_SIZE,
    build_piece_detector,
    select_detector_device,
)

LOGGER = logging.getLogger(__name__)
_TRAIN_LOG_EVERY = 10
_DEFAULT_PATIENCE = 5


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]],
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            total_loss += float(sum(loss_dict.values()).item()) * len(images)
            total_samples += len(images)
    return total_loss / total_samples if total_samples else 0.0


def train_detector(
    manifest_path: Path,
    output_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    patience: int,
) -> None:
    """Train a detector on the prepared board-image dataset."""
    device = select_detector_device()
    train_ds = DetectionDataset(
        manifest_path,
        split="train",
        image_size=image_size,
        augment=True,
    )
    val_ds = DetectionDataset(
        manifest_path,
        split="val",
        image_size=image_size,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=2,
    )
    LOGGER.info(
        f"Training detector manifest={manifest_path} output={output_path} "
        f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"image_size={image_size} batch_size={batch_size}"
    )

    model = build_piece_detector().to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    params = {
        "dataset_manifest": str(manifest_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "patience": patience,
        "augmentation": "flip,color_jitter,scale_jitter,blur",
        "architecture": DETECTOR_ARCHITECTURE,
    }
    with training_run(EXPERIMENTS["piece"], "piece-detector-train", params):
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for step, (images, targets) in enumerate(train_dl, start=1):
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(images)
                if step == 1 or step % _TRAIN_LOG_EVERY == 0:
                    LOGGER.info(
                        f"Detector epoch={epoch}/{epochs} "
                        f"step={step}/{len(train_dl)} loss={loss.item():.4f}"
                    )

            train_loss = epoch_loss / len(train_ds) if len(train_ds) else 0.0
            val_loss = _evaluate_loss(model, val_dl, device)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": round(current_lr, 8),
            }
            log_epoch_metrics(metrics, epoch)
            LOGGER.info(
                f"Detector epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={current_lr:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(output_path))
                LOGGER.info(f"Saved improved detector checkpoint to {output_path}")
                continue

            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info(
                    f"Detector early stopping after {epoch} epochs "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

        if output_path.exists():
            log_artifact(output_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the chess piece detector.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/piece_detector.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=_DEFAULT_PATIENCE)
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        dest="image_size",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    train_detector(
        args.manifest,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
