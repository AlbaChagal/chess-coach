"""Automated hyperparameter search using Optuna + MLflow.

Runs a Bayesian HP search over the training pipeline, logging every trial as a
nested MLflow child run under a parent study run.

Usage::

    uv run python scripts/hp_search.py \\
        --squares data/chesscog/squares.csv \\
        --trials  30 \\
        --output  models/ \\
        --model   piece      # or "occupancy" or "both"

Search space:
- ``lr``: log-uniform [1e-5, 1e-2]
- ``batch_size``: categorical [32, 64, 128]
- ``patience``: int [3, 10]
- ``color_jitter``: uniform [0.1, 0.4]
- ``rotation_deg``: int [3, 15]

MLflow structure::

    chesscoach-hp-search/
      hp-search-<model>-<date>   ← parent run (study params + best result)
        trial-000                ← nested child run per Optuna trial
        trial-001
        ...
"""

from __future__ import annotations

import argparse
import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlflow
import optuna
from torchvision import transforms

from chesscoach import mlops
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.torch_utils import select_device
from chesscoach.vision.types import PIECE_LABELS
from scripts.train import (
    SquareDataset,
    _EVAL_TRANSFORM,
    _INPUT_SIZE,
    _PIECE_LABELS_NO_EMPTY,
    train_model,
)

ModelChoice = Literal["occupancy", "piece", "both"]

LOGGER = logging.getLogger(__name__)


def _build_train_transform(color_jitter: float, rotation_deg: int) -> transforms.Compose:
    """Build a training transform with the given augmentation parameters."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter),
            transforms.RandomRotation(rotation_deg),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_occupancy_label_map() -> dict[str, int]:
    label_map = {"empty": 0}
    for lbl in PIECE_LABELS:
        if lbl != "empty":
            label_map[lbl] = 1
    return label_map


def _run_trial(
    trial: optuna.Trial,
    squares_csv: Path,
    output_dir: Path,
    model_choice: ModelChoice,
    max_epochs: int,
    trial_idx: int,
) -> float:
    """Train one trial and return the best validation loss (objective to minimise)."""
    lr: float = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size: int = trial.suggest_categorical("batch_size", [32, 64, 128])  # type: ignore[assignment]
    patience: int = trial.suggest_int("patience", 3, 10)
    color_jitter: float = trial.suggest_float("color_jitter", 0.1, 0.4)
    rotation_deg: int = trial.suggest_int("rotation_deg", 3, 15)

    root = squares_csv.parent
    device = select_device()
    trial_name = f"trial-{trial_idx:03d}"

    trial_params: dict[str, object] = {
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "color_jitter": color_jitter,
        "rotation_deg": rotation_deg,
        "model_choice": model_choice,
    }

    train_transform = _build_train_transform(color_jitter, rotation_deg)
    trial_dir = output_dir / f"trial_{trial_idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    with mlops.training_run(
        mlops.EXPERIMENTS["hp_search"],
        trial_name,
        trial_params,
        nested=True,
    ):
        if model_choice in ("occupancy", "both"):
            occ_label_map = _make_occupancy_label_map()

            # Patch SquareDataset to use the trial's transform
            original_init = SquareDataset.__init__

            def patched_init(
                self: SquareDataset,
                csv_path: Path,
                split: object,
                label_map: dict[str, int],
                transform: transforms.Compose,
                root_dir: Path,
            ) -> None:
                if transform is not _EVAL_TRANSFORM:
                    transform = train_transform
                original_init(self, csv_path, split, label_map, transform, root_dir)  # type: ignore[arg-type]

            SquareDataset.__init__ = patched_init  # type: ignore[method-assign]
            try:
                occ_history = train_model(
                    csv_path=squares_csv,
                    root=root,
                    label_map=occ_label_map,
                    num_classes=2,
                    max_epochs=max_epochs,
                    output_path=trial_dir / "occupancy.pt",
                    device=device,
                    model_name=f"Occupancy[trial={trial_idx}]",
                    class_names=["empty", "occupied"],
                    learning_rate=lr,
                    image_column="occupancy_image_path",
                    patience=patience,
                    on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
                )
            finally:
                SquareDataset.__init__ = original_init  # type: ignore[method-assign]

            occ_best = min(occ_history["val_loss"]) if occ_history["val_loss"] else float("inf")
            best_val_loss = min(best_val_loss, occ_best)

        if model_choice in ("piece", "both"):
            piece_label_map = {lbl: idx for idx, lbl in enumerate(_PIECE_LABELS_NO_EMPTY)}
            piece_class_names = list(_PIECE_LABELS_NO_EMPTY)

            original_init = SquareDataset.__init__

            def patched_init(  # type: ignore[no-redef]
                self: SquareDataset,
                csv_path: Path,
                split: object,
                label_map: dict[str, int],
                transform: transforms.Compose,
                root_dir: Path,
            ) -> None:
                if transform is not _EVAL_TRANSFORM:
                    transform = train_transform
                original_init(self, csv_path, split, label_map, transform, root_dir)  # type: ignore[arg-type]

            SquareDataset.__init__ = patched_init  # type: ignore[method-assign]
            try:
                piece_history = train_model(
                    csv_path=squares_csv,
                    root=root,
                    label_map=piece_label_map,
                    num_classes=12,
                    max_epochs=max_epochs,
                    output_path=trial_dir / "piece.pt",
                    device=device,
                    model_name=f"Piece[trial={trial_idx}]",
                    class_names=piece_class_names,  # type: ignore[arg-type]
                    learning_rate=lr,
                    class_weighted_loss=True,
                    upsample_minority_classes=True,
                    image_column="piece_image_path",
                    patience=patience,
                    on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
                )
            finally:
                SquareDataset.__init__ = original_init  # type: ignore[method-assign]

            piece_best = min(piece_history["val_loss"]) if piece_history["val_loss"] else float("inf")
            best_val_loss = min(best_val_loss, piece_best)

    return best_val_loss


def run_hp_search(
    squares_csv: Path,
    output_dir: Path,
    model_choice: ModelChoice,
    n_trials: int,
    max_epochs: int,
) -> None:
    """Run an Optuna hyperparameter study and register the best checkpoint.

    Args:
        squares_csv: Path to the squares.csv manifest.
        output_dir: Directory to save trial checkpoints.
        model_choice: Which model(s) to optimise: ``"occupancy"``, ``"piece"``, or ``"both"``.
        n_trials: Number of Optuna trials.
        max_epochs: Max epochs per trial (keep low for search, e.g. 5–10).
    """
    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    study_run_name = f"hp-search-{model_choice}-{date_tag}"

    study_params: dict[str, object] = {
        "model_choice": model_choice,
        "n_trials": n_trials,
        "max_epochs": max_epochs,
        "dataset_csv": str(squares_csv),
    }

    mlflow.set_experiment(mlops.EXPERIMENTS["hp_search"])
    with mlops.training_run(mlops.EXPERIMENTS["hp_search"], study_run_name, study_params) as parent_run:
        trial_counter = 0

        def objective(trial: optuna.Trial) -> float:
            nonlocal trial_counter
            idx = trial_counter
            trial_counter += 1
            return _run_trial(trial, squares_csv, output_dir, model_choice, max_epochs, idx)

        study = optuna.create_study(
            direction="minimize",
            study_name=study_run_name,
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        LOGGER.info(
            f"HP search complete best_trial={best.number} "
            f"best_val_loss={best.value:.5f} best_params={best.params}"
        )

        # Log best trial summary on the parent run
        mlflow.log_params({f"best_{k}": str(v) for k, v in best.params.items()})
        mlflow.log_metric("best_val_loss", best.value or float("inf"))

        # Save study object as artifact
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            study_pkl_path = Path(tmp.name)
        study_pkl_path.write_bytes(pickle.dumps(study))
        mlops.log_artifact(study_pkl_path)
        study_pkl_path.unlink(missing_ok=True)

        # Save HP importance plot
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization.matplotlib as optuna_vis

            optuna_vis.plot_param_importances(study)
            importance_path = output_dir / "hp_importance.png"
            plt.savefig(str(importance_path), bbox_inches="tight")
            plt.close()
            mlops.log_artifact(importance_path)
            LOGGER.info(f"HP importance plot saved to {importance_path}")
        except Exception as exc:
            LOGGER.warning(f"Could not generate HP importance plot: {exc}")

        # Register best checkpoint(s)
        best_trial_dir = output_dir / f"trial_{best.number:03d}"
        parent_run_id = parent_run.info.run_id

        if model_choice in ("occupancy", "both"):
            occ_ckpt = best_trial_dir / "occupancy.pt"
            if occ_ckpt.exists():
                # Re-open parent run to register (we're already in it)
                mlops.register_checkpoint(occ_ckpt, "OccupancyClassifier")
                LOGGER.info(
                    f"Registered best occupancy checkpoint run_id={parent_run_id}"
                )

        if model_choice in ("piece", "both"):
            piece_ckpt = best_trial_dir / "piece.pt"
            if piece_ckpt.exists():
                mlops.register_checkpoint(piece_ckpt, "PieceClassifier")
                LOGGER.info(
                    f"Registered best piece checkpoint run_id={parent_run_id}"
                )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for chesscoach classifiers."
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
        default=Path("models/hp_search"),
        help="Directory to save trial checkpoints (default: models/hp_search/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["occupancy", "piece", "both"],
        default="piece",
        dest="model",
        help="Which model(s) to optimise (default: piece).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        dest="max_epochs",
        help="Max epochs per trial (default: 10).",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    args.output.mkdir(parents=True, exist_ok=True)
    run_hp_search(
        squares_csv=args.squares,
        output_dir=args.output,
        model_choice=args.model,  # type: ignore[arg-type]
        n_trials=args.trials,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
