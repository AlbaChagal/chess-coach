"""Train a learned board-corner localizer on raw board images."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.mlops import EXPERIMENTS, log_artifact, log_epoch_metrics, training_run
from chesscoach.vision.board_localizer import (
    BOARD_LOCALIZER_ARCHITECTURE,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
    build_board_localizer,
    select_board_localizer_device,
)
from chesscoach.vision.board_localizer_dataset import BoardLocalizationDataset

try:
    from scripts.prepare_board_localizer_dataset import prepare_board_localizer_dataset
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.prepare_board_localizer_dataset import prepare_board_localizer_dataset

LOGGER = logging.getLogger(__name__)
_TRAIN_LOG_EVERY = 10
_DEFAULT_PATIENCE = 5
_EDGE_LENGTH_LOSS_WEIGHT = 1.0
_AREA_LOSS_WEIGHT = 0.5


def _resolve_manifest_path(
    *,
    manifest_path: Path | None,
    raw_input: Path | None,
    prepared_output: Path | None,
) -> Path:
    """Return the board-localizer manifest path, preparing it if needed."""
    if manifest_path is not None:
        return manifest_path
    if raw_input is None:
        raise ValueError("Either manifest_path or raw_input must be provided.")

    output_dir = prepared_output or Path("data/chess_boards/board_localizer")
    return prepare_board_localizer_dataset(raw_input, output_dir)


def _pixel_corner_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Return mean per-sample corner error in original-image pixels."""
    pred_corners = predictions.view(-1, 4, 2)
    target_corners = targets.view(-1, 4, 2)
    scales = sizes.unsqueeze(1)
    deltas = (pred_corners - target_corners) * scales
    return deltas.pow(2).sum(dim=2).sqrt().mean(dim=1)


def _edge_length_features(corners: torch.Tensor) -> torch.Tensor:
    """Return ordered board-edge lengths for flattened corner tensors."""
    ordered_corners = corners.view(-1, 4, 2)
    edge_starts = ordered_corners
    edge_ends = torch.roll(ordered_corners, shifts=-1, dims=1)
    return (edge_starts - edge_ends).pow(2).sum(dim=2).sqrt()


def _quadrilateral_area(corners: torch.Tensor) -> torch.Tensor:
    """Return quadrilateral areas for flattened corner tensors."""
    ordered_corners = corners.view(-1, 4, 2)
    x = ordered_corners[:, :, 0]
    y = ordered_corners[:, :, 1]
    cross_terms = x * torch.roll(y, shifts=-1, dims=1)
    reverse_cross_terms = y * torch.roll(x, shifts=-1, dims=1)
    return 0.5 * torch.abs((cross_terms - reverse_cross_terms).sum(dim=1))


def _board_localizer_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Return the composite localizer loss terms."""
    corner_loss = criterion(predictions, targets)
    predicted_edges = _edge_length_features(predictions)
    target_edges = _edge_length_features(targets)
    edge_loss = torch.nn.functional.smooth_l1_loss(predicted_edges, target_edges)
    predicted_area = _quadrilateral_area(predictions)
    target_area = _quadrilateral_area(targets)
    area_loss = torch.nn.functional.smooth_l1_loss(predicted_area, target_area)
    total_loss = (
        corner_loss
        + _EDGE_LENGTH_LOSS_WEIGHT * edge_loss
        + _AREA_LOSS_WEIGHT * area_loss
    )
    return {
        "total": total_loss,
        "corner": corner_loss,
        "edge": edge_loss,
        "area": area_loss,
    }


def _board_localizer_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Return the scalar composite loss used for localizer training."""
    return _board_localizer_loss_components(predictions, targets, criterion)["total"]


def _evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_mean_corner_error_px = 0.0
    total_leq_20px = 0
    total_samples = 0
    with torch.no_grad():
        for images, targets, sizes in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            sizes = sizes.to(device)
            predictions = model(images)
            loss = _board_localizer_loss(predictions, targets, criterion)
            sample_errors_px = _pixel_corner_error(predictions, targets, sizes)
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_mean_corner_error_px += float(sample_errors_px.sum().item())
            total_leq_20px += int((sample_errors_px <= 20.0).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / total_samples,
        total_mean_corner_error_px / total_samples,
        total_leq_20px / total_samples,
    )


def _load_sample_weights(
    dataset: BoardLocalizationDataset,
    weights_path: Path | None,
) -> list[float] | None:
    """Load per-sample weights keyed by dataset image path."""
    if weights_path is None:
        return None

    payload = json.loads(weights_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid hard-example weights file: {weights_path}")

    default_weight = payload.get("default_weight", 1.0)
    if not isinstance(default_weight, int | float):
        raise ValueError(
            f"Invalid default_weight in hard-example weights file: {weights_path}"
        )

    samples = payload.get("samples", payload)
    if not isinstance(samples, dict):
        raise ValueError(f"Invalid samples mapping in hard-example weights file: {weights_path}")

    weights: list[float] = []
    matched = 0
    for sample_id in dataset.sample_ids():
        raw_weight = samples.get(sample_id, default_weight)
        if sample_id in samples:
            matched += 1
        if not isinstance(raw_weight, int | float):
            raise ValueError(
                f"Invalid sample weight for {sample_id} in hard-example weights file"
            )
        weights.append(max(float(raw_weight), 1e-3))

    LOGGER.info(
        f"Loaded hard-example weights from {weights_path} "
        f"matched_samples={matched}/{len(weights)}"
    )
    return weights


def train_board_localizer(
    manifest_path: Path | None,
    output_path: Path,
    *,
    raw_input: Path | None,
    prepared_output: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    patience: int,
    hard_example_weights: Path | None,
) -> None:
    """Train a board-corner localizer on the prepared raw-image manifest."""
    resolved_manifest_path = _resolve_manifest_path(
        manifest_path=manifest_path,
        raw_input=raw_input,
        prepared_output=prepared_output,
    )
    device = select_board_localizer_device()
    train_ds = BoardLocalizationDataset(
        resolved_manifest_path,
        split="train",
        image_size=image_size,
        augment=True,
    )
    val_ds = BoardLocalizationDataset(
        resolved_manifest_path,
        split="val",
        image_size=image_size,
    )
    sample_weights = _load_sample_weights(train_ds, hard_example_weights)
    train_sampler = (
        WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        if sample_weights is not None
        else None
    )
    train_dl: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
    )
    val_dl: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    LOGGER.info(
        f"Training board localizer manifest={resolved_manifest_path} output={output_path} "
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
    best_val_mean_corner_error_px = float("inf")
    epochs_without_improvement = 0

    params = {
        "dataset_manifest": str(resolved_manifest_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "patience": patience,
        "augmentation": "perspective_jitter,color_jitter,blur",
        "architecture": BOARD_LOCALIZER_ARCHITECTURE,
        "selection_metric": "val_mean_corner_error_px",
        "edge_length_loss_weight": _EDGE_LENGTH_LOSS_WEIGHT,
        "area_loss_weight": _AREA_LOSS_WEIGHT,
        "hard_example_weights": str(hard_example_weights)
        if hard_example_weights is not None
        else "",
    }
    with training_run(EXPERIMENTS["piece"], "board-localizer-train", params):
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_corner_loss = 0.0
            epoch_edge_loss = 0.0
            epoch_area_loss = 0.0
            for step, (images, targets, _sizes) in enumerate(train_dl, start=1):
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                predictions = model(images)
                loss_components = _board_localizer_loss_components(
                    predictions,
                    targets,
                    criterion,
                )
                loss = loss_components["total"]
                loss.backward()
                optimizer.step()
                batch_size = images.shape[0]
                epoch_loss += float(loss.item()) * batch_size
                epoch_corner_loss += float(loss_components["corner"].item()) * batch_size
                epoch_edge_loss += float(loss_components["edge"].item()) * batch_size
                epoch_area_loss += float(loss_components["area"].item()) * batch_size
                if step == 1 or step % _TRAIN_LOG_EVERY == 0:
                    LOGGER.info(
                        f"Board localizer epoch={epoch}/{epochs} "
                        f"step={step}/{len(train_dl)} loss={loss.item():.4f} "
                        f"corner={loss_components['corner'].item():.4f} "
                        f"edge={loss_components['edge'].item():.4f} "
                        f"area={loss_components['area'].item():.4f}"
                    )

            train_loss = epoch_loss / len(train_ds) if len(train_ds) else 0.0
            train_corner_loss = (
                epoch_corner_loss / len(train_ds) if len(train_ds) else 0.0
            )
            train_edge_loss = epoch_edge_loss / len(train_ds) if len(train_ds) else 0.0
            train_area_loss = epoch_area_loss / len(train_ds) if len(train_ds) else 0.0
            val_loss, val_mean_corner_error_px, val_boards_leq_20px = _evaluate_model(
                model,
                val_dl,
                device,
                criterion,
            )
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "train_loss": round(train_loss, 5),
                "train_corner_loss": round(train_corner_loss, 5),
                "train_edge_loss": round(train_edge_loss, 5),
                "train_area_loss": round(train_area_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_mean_corner_error_px": round(val_mean_corner_error_px, 5),
                "val_boards_leq_20px": round(val_boards_leq_20px, 5),
                "lr": round(current_lr, 8),
            }
            log_epoch_metrics(metrics, epoch)
            LOGGER.info(
                f"Board localizer epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"train_corner_loss={train_corner_loss:.4f} "
                f"train_edge_loss={train_edge_loss:.4f} "
                f"train_area_loss={train_area_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_mean_corner_error_px={val_mean_corner_error_px:.2f} "
                f"val_boards_leq_20px={val_boards_leq_20px:.4f} "
                f"lr={current_lr:.6f}"
            )

            if val_mean_corner_error_px < best_val_mean_corner_error_px:
                best_val_mean_corner_error_px = val_mean_corner_error_px
                epochs_without_improvement = 0
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(output_path))
                LOGGER.info(f"Saved improved board localizer checkpoint to {output_path}")
                continue

            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info(
                    f"Board localizer early stopping after {epoch} epochs "
                    f"best_val_mean_corner_error_px={best_val_mean_corner_error_px:.2f}"
                )
                break

        if output_path.exists():
            log_artifact(output_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the board localizer.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=None,
        dest="raw_input",
        help="Optional raw split directory to prepare into a board-localizer manifest.",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=None,
        dest="prepared_output",
        help="Optional output directory for manifests prepared from raw input.",
    )
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
        "--hard-example-weights",
        type=Path,
        default=None,
        dest="hard_example_weights",
        help="Optional JSON file mapping image_path to sample weight.",
    )
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
        raw_input=args.raw_input,
        prepared_output=args.prepared_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        patience=args.patience,
        hard_example_weights=args.hard_example_weights,
    )


if __name__ == "__main__":
    main()
