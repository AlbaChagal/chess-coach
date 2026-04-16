"""Adapt trained models to a specific user's piece set via transfer learning.

Takes 2 photographs of a chess board in the **starting position** (known FEN),
extracts the 128 labeled squares, and fine-tunes only the final fully-connected
layer of each model for 5 epochs.

This produces custom checkpoints that are highly accurate on the user's
specific pieces and board under their lighting conditions.

Usage::

    uv run python scripts/transfer_learn.py \\
        --photo1 starting_pos_1.jpg \\
        --photo2 starting_pos_2.jpg \\
        --occupancy-model models/occupancy.pt \\
        --piece-model     models/piece.pt \\
        --output          models/custom/

Expected accuracy gain: 5–10% board accuracy on the user's specific piece set.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from chesscoach import mlops
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.torch_utils import select_device
from chesscoach.vision.board_detector import detect_board, split_into_squares
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
_INPUT_SIZE = 100
_FINETUNE_EPOCHS = 5
_LR = 1e-4

_PIECE_LABELS_NO_EMPTY: list[PieceLabel] = [lbl for lbl in PIECE_LABELS if lbl != "empty"]

_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# FEN character → PieceLabel
_FEN_CHAR_TO_LABEL: dict[str, PieceLabel] = {
    "P": "wP", "N": "wN", "B": "wB", "R": "wR", "Q": "wQ", "K": "wK",
    "p": "bP", "n": "bN", "b": "bB", "r": "bR", "q": "bQ", "k": "bK",
}
LOGGER = logging.getLogger(__name__)


def _fen_to_labels(fen_placement: str) -> list[PieceLabel]:
    """Expand FEN placement into a flat 64-element list of PieceLabels."""
    labels: list[PieceLabel] = []
    for rank in fen_placement.split("/"):
        for ch in rank:
            if ch.isdigit():
                labels.extend(["empty"] * int(ch))  # type: ignore[arg-type]
            else:
                labels.append(_FEN_CHAR_TO_LABEL[ch])
    return labels


def _extract_squares(photo_path: Path) -> list[tuple[PieceLabel, torch.Tensor]]:
    """Detect the board in a photo and return (label, tensor) pairs."""
    LOGGER.debug(f"Extracting labeled squares from {photo_path}")
    bgr = cv2.imread(str(photo_path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {photo_path}")

    warped = detect_board(bgr)
    square_grid = split_into_squares(warped)
    labels = _fen_to_labels(_STARTING_FEN)

    samples: list[tuple[PieceLabel, torch.Tensor]] = []
    flat_squares = [sq for row in square_grid for sq in row]
    for sq, label in zip(flat_squares, labels):
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = _TRANSFORM(rgb)  # type: ignore[assignment]
        samples.append((label, tensor))
    return samples


def _load_model(checkpoint: Path, num_classes: int, device: torch.device) -> nn.Module:
    LOGGER.info(
        f"Loading transfer-learning model architecture=ResNet18 "
        f"checkpoint={checkpoint} num_classes={num_classes} device={device}"
    )
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def _finetune(
    model: nn.Module,
    tensors: torch.Tensor,
    class_indices: torch.Tensor,
    device: torch.device,
    output_path: Path,
    metric_key: str = "loss",
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> None:
    """Fine-tune the final FC layer only and save the checkpoint."""
    # Freeze all layers except the final FC
    for param in model.parameters():
        param.requires_grad = False
    fc: nn.Linear = model.fc  # type: ignore[assignment]
    for param in fc.parameters():
        param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(fc.parameters(), lr=_LR)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(tensors, class_indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    LOGGER.info(
        f"Fine-tuning model={model.__class__.__name__} samples={len(dataset)} "
        f"batch_size=32 output={output_path}"
    )

    for epoch in range(1, _FINETUNE_EPOCHS + 1):
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(images)
            if LOGGER.isEnabledFor(logging.DEBUG) and step % 100 == 0:
                LOGGER.debug(
                    f"Transfer fine-tune epoch={epoch} step={step}/{len(loader)} "
                    f"batch_loss={loss.item():.4f}"
                )
        avg_loss = epoch_loss / len(dataset)
        LOGGER.info(
            f"Transfer fine-tune epoch {epoch}/{_FINETUNE_EPOCHS} "
            f"loss={avg_loss:.4f}"
        )
        if on_epoch is not None:
            on_epoch(epoch, {metric_key: round(avg_loss, 5)})

    # Restore all params to requires_grad=True before saving
    for param in model.parameters():
        param.requires_grad = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))
    LOGGER.info(f"Saved fine-tuned checkpoint to {output_path}")


def transfer_learn(
    photo1: Path,
    photo2: Path,
    occupancy_model_path: Path,
    piece_model_path: Path,
    output_dir: Path,
    occupancy_run_id: str | None = None,
    piece_run_id: str | None = None,
) -> None:
    """Fine-tune both models on 2 starting-position photos.

    Args:
        photo1: First photo of the board in starting position.
        photo2: Second photo of the board in starting position.
        occupancy_model_path: Pre-trained occupancy checkpoint.
        piece_model_path: Pre-trained piece checkpoint.
        output_dir: Directory to save the custom checkpoints.
        occupancy_run_id: Optional MLflow run ID of the base occupancy model.
        piece_run_id: Optional MLflow run ID of the base piece model.
    """
    device = select_device()
    LOGGER.info(
        f"Starting transfer learning photo1={photo1} photo2={photo2} "
        f"occupancy_model={occupancy_model_path} piece_model={piece_model_path} "
        f"output={output_dir} device={device}"
    )

    LOGGER.info("Extracting squares from photos")
    samples: list[tuple[PieceLabel, torch.Tensor]] = []
    for photo in (photo1, photo2):
        LOGGER.info(f"Processing photo {photo.name}")
        samples.extend(_extract_squares(photo))

    LOGGER.info(f"Extracted {len(samples)} labeled squares")

    # Build tensors
    all_tensors = torch.stack([t for _, t in samples])
    all_labels_str = [lbl for lbl, _ in samples]

    params: dict[str, object] = {
        "photo1": str(photo1),
        "photo2": str(photo2),
        "occupancy_source": str(occupancy_model_path),
        "piece_source": str(piece_model_path),
        "finetune_epochs": _FINETUNE_EPOCHS,
        "lr": _LR,
        "n_squares": len(samples),
    }

    with mlops.training_run(mlops.EXPERIMENTS["transfer"], "transfer-learn", params) as run:
        import mlflow as _mlflow

        if occupancy_run_id:
            _mlflow.set_tag("source_occupancy_run", occupancy_run_id)
        if piece_run_id:
            _mlflow.set_tag("source_piece_run", piece_run_id)
        _ = run  # keep reference

        # --- Occupancy fine-tuning ---
        LOGGER.info("=== Fine-tuning occupancy model ===")
        occ_model = _load_model(occupancy_model_path, 2, device)
        occ_indices = torch.tensor(
            [0 if lbl == "empty" else 1 for lbl in all_labels_str], dtype=torch.long
        )
        occ_out = output_dir / "occupancy.pt"
        _finetune(
            occ_model,
            all_tensors,
            occ_indices,
            device,
            occ_out,
            metric_key="occ_loss",
            on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
        )
        mlops.log_artifact(occ_out)

        # --- Piece fine-tuning (non-empty squares only) ---
        LOGGER.info("=== Fine-tuning piece model ===")
        piece_model = _load_model(piece_model_path, 12, device)
        piece_mask = [i for i, lbl in enumerate(all_labels_str) if lbl != "empty"]
        if not piece_mask:
            LOGGER.warning("No occupied squares found; skipping piece model fine-tuning")
        else:
            piece_tensors = all_tensors[piece_mask]
            piece_indices = torch.tensor(
                [_PIECE_LABELS_NO_EMPTY.index(all_labels_str[i]) for i in piece_mask],  # type: ignore[arg-type]
                dtype=torch.long,
            )
            piece_out = output_dir / "piece.pt"
            _finetune(
                piece_model,
                piece_tensors,
                piece_indices,
                device,
                piece_out,
                metric_key="piece_loss",
                on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
            )
            mlops.log_artifact(piece_out)

    LOGGER.info("Transfer learning complete")
    LOGGER.info(f"Custom checkpoints saved to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adapt models to a specific piece set using 2 starting-position photos."
    )
    add_logging_args(parser)
    parser.add_argument("--photo1", type=Path, required=True)
    parser.add_argument("--photo2", type=Path, required=True)
    parser.add_argument(
        "--occupancy-model", type=Path, required=True, dest="occupancy_model"
    )
    parser.add_argument(
        "--piece-model", type=Path, required=True, dest="piece_model"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/custom"), dest="output"
    )
    parser.add_argument(
        "--occupancy-run-id",
        type=str,
        default=None,
        dest="occupancy_run_id",
        help="MLflow run ID of the base occupancy model (for lineage tracking).",
    )
    parser.add_argument(
        "--piece-run-id",
        type=str,
        default=None,
        dest="piece_run_id",
        help="MLflow run ID of the base piece model (for lineage tracking).",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    transfer_learn(
        photo1=args.photo1,
        photo2=args.photo2,
        occupancy_model_path=args.occupancy_model,
        piece_model_path=args.piece_model,
        output_dir=args.output,
        occupancy_run_id=args.occupancy_run_id,
        piece_run_id=args.piece_run_id,
    )


if __name__ == "__main__":
    main()
