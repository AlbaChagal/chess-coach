"""Train a learned board-corner localizer on raw board images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.mlops import EXPERIMENTS, log_artifact, log_epoch_metrics, training_run
from chesscoach.vision.board_localizer import (
    BOARD_LOCALIZER_ARCHITECTURE,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
    build_board_localizer,
    select_board_localizer_device,
)
from chesscoach.vision.board_localizer_dataset import BoardLocalizationDataset

LOGGER = logging.getLogger(__name__)
_TRAIN_LOG_EVERY = 10
_DEFAULT_PATIENCE = 5


def _evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mean_corner_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            sample_errors = (
                (predictions.view(-1, 4, 2) - targets.view(-1, 4, 2))
                .pow(2)
                .sum(dim=2)
                .sqrt()
                .mean(dim=1)
            )
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mean_corner_error += float(sample_errors.sum().item())
            total_samples += batch_size
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_mean_corner_error / total_samples


def train_board_localizer(
    manifest_path: Path,
    output_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    patience: int,
) -> None:
    """Train a board-corner localizer on the prepared raw-image manifest."""
    device = select_board_localizer_device()
    train_ds = BoardLocalizationDataset(
        manifest_path,
        split="train",
        image_size=image_size,
        augment=True,
    )
    val_ds = BoardLocalizationDataset(
        manifest_path,
        split="val",
        image_size=image_size,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    LOGGER.info(
        f"Training board localizer manifest={manifest_path} output={output_path} "
        f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"image_size={image_size} batch_size={batch_size}"
    )

    model = build_board_localizer().to(device)
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
    criterion = torch.nn.SmoothL1Loss()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    params = {
        "dataset_manifest": str(manifest_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "patience": patience,
        "augmentation": "perspective_jitter,color_jitter,blur",
        "architecture": BOARD_LOCALIZER_ARCHITECTURE,
    }
    with training_run(EXPERIMENTS["piece"], "board-localizer-train", params):
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for step, (images, targets) in enumerate(train_dl, start=1):
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * images.shape[0]
                if step == 1 or step % _TRAIN_LOG_EVERY == 0:
                    LOGGER.info(
                        f"Board localizer epoch={epoch}/{epochs} "
                        f"step={step}/{len(train_dl)} loss={loss.item():.4f}"
                    )

            train_loss = epoch_loss / len(train_ds) if len(train_ds) else 0.0
            val_loss, val_mean_corner_error = _evaluate_model(
                model,
                val_dl,
                device,
                criterion,
            )
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_mean_corner_error_norm": round(val_mean_corner_error, 5),
                "lr": round(current_lr, 8),
            }
            log_epoch_metrics(metrics, epoch)
            LOGGER.info(
                f"Board localizer epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mean_corner_error_norm={val_mean_corner_error:.4f} "
                f"lr={current_lr:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(output_path))
                LOGGER.info(f"Saved improved board localizer checkpoint to {output_path}")
                continue

            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info(
                    f"Board localizer early stopping after {epoch} epochs "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

        if output_path.exists():
            log_artifact(output_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the board localizer.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/board_localizer.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=_DEFAULT_PATIENCE)
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="image_size",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    train_board_localizer(
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
