"""Run a fixed list of hyperparameter configurations sequentially.

Unlike ``hp_search.py`` (Optuna Bayesian search), this script trains one model
per config in the order you specify — no surprises, easy to leave running
overnight.

Usage::

    uv run python scripts/hyperparam_search.py \\
        --squares data/chesscog/squares.csv \\
        --output  models/search \\
        --configs configs/overnight.json

Config file — a JSON list of objects, one per run::

    [
      {"name": "baseline",     "lr": 3e-4, "patience": 5,  "max_epochs": 20},
      {"name": "high-lr",      "lr": 1e-3, "patience": 5,  "max_epochs": 20},
      {"name": "low-lr",       "lr": 1e-5, "patience": 10, "max_epochs": 30},
      {"name": "small-batch",  "lr": 3e-4, "batch_size": 32, "max_epochs": 20},
      {"name": "large-batch",  "lr": 3e-4, "batch_size": 128, "max_epochs": 20}
    ]

Supported keys per config entry (all optional except none are required — defaults are used for anything omitted):

  name        Human-readable label shown in MLflow (auto-generated if omitted)
  model       "occupancy", "piece", or "both"  (default: "both")
  lr          Learning rate                     (default: 3e-4)
  batch_size  Mini-batch size                   (default: 64)
  patience    Early-stopping patience           (default: 5)
  max_epochs  Maximum training epochs           (default: 20)

Each config produces one MLflow run per model in the matching experiment
(``chesscoach-occupancy`` / ``chesscoach-piece``).  A summary table is
printed when all configs finish.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from chesscoach import mlops
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.torch_utils import select_device
from chesscoach.vision.types import PIECE_LABELS
from scripts.train import (
    _BATCH_SIZE,
    _PIECE_LABELS_NO_EMPTY,
    train_model,
)

LOGGER = logging.getLogger(__name__)

ModelChoice = Literal["occupancy", "piece", "both"]

_DEFAULT_LR = 3e-4
_DEFAULT_PATIENCE = 5
_DEFAULT_MAX_EPOCHS = 20
_DEFAULT_MODEL: ModelChoice = "both"


@dataclass
class RunConfig:
    """A single hyperparameter configuration."""

    name: str
    model: ModelChoice
    lr: float
    batch_size: int
    patience: int
    max_epochs: int


@dataclass
class RunResult:
    """Outcome of one config run."""

    name: str
    model: ModelChoice
    best_occ_val_loss: float = float("inf")
    best_piece_val_loss: float = float("inf")
    error: str | None = None
    mlflow_run_ids: list[str] = field(default_factory=list)


def _parse_configs(path: Path) -> list[RunConfig]:
    """Load and validate a JSON config file into :class:`RunConfig` objects."""
    raw: list[dict[str, object]] = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Config file must be a JSON list, got {type(raw).__name__}")

    configs: list[RunConfig] = []
    for i, entry in enumerate(raw):
        name = str(entry.get("name", f"run-{i:03d}"))
        model_raw = str(entry.get("model", _DEFAULT_MODEL))
        if model_raw not in ("occupancy", "piece", "both"):
            raise ValueError(f"Config {name!r}: 'model' must be 'occupancy', 'piece', or 'both'")
        configs.append(
            RunConfig(
                name=name,
                model=model_raw,  # type: ignore[arg-type]
                lr=float(entry.get("lr", _DEFAULT_LR)),  # type: ignore[arg-type]
                batch_size=int(entry.get("batch_size", _BATCH_SIZE)),  # type: ignore[arg-type]
                patience=int(entry.get("patience", _DEFAULT_PATIENCE)),  # type: ignore[arg-type]
                max_epochs=int(entry.get("max_epochs", _DEFAULT_MAX_EPOCHS)),  # type: ignore[arg-type]
            )
        )
    return configs


def _make_occupancy_label_map() -> dict[str, int]:
    label_map = {"empty": 0}
    for lbl in PIECE_LABELS:
        if lbl != "empty":
            label_map[lbl] = 1
    return label_map


def _train_one(
    cfg: RunConfig,
    squares_csv: Path,
    output_dir: Path,
) -> RunResult:
    """Train all models for a single :class:`RunConfig`."""
    result = RunResult(name=cfg.name, model=cfg.model)
    root = squares_csv.parent
    device = select_device()
    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        f"=== Starting config name={cfg.name} model={cfg.model} "
        f"lr={cfg.lr} batch_size={cfg.batch_size} "
        f"patience={cfg.patience} max_epochs={cfg.max_epochs} ==="
    )

    shared_params: dict[str, object] = {
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "patience": cfg.patience,
        "max_epochs": cfg.max_epochs,
        "architecture": "ResNet18",
        "dataset_csv": str(squares_csv),
        "config_name": cfg.name,
    }

    try:
        if cfg.model in ("occupancy", "both"):
            occ_checkpoint = run_dir / "occupancy.pt"
            with mlops.training_run(
                mlops.EXPERIMENTS["occupancy"],
                cfg.name,
                shared_params,
            ) as run:
                history = train_model(
                    csv_path=squares_csv,
                    root=root,
                    label_map=_make_occupancy_label_map(),
                    num_classes=2,
                    max_epochs=cfg.max_epochs,
                    output_path=occ_checkpoint,
                    device=device,
                    model_name=f"Occupancy[{cfg.name}]",
                    class_names=["empty", "occupied"],
                    learning_rate=cfg.lr,
                    image_column="occupancy_image_path",
                    patience=cfg.patience,
                    batch_size=cfg.batch_size,
                    on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
                )
                mlops.log_artifact(occ_checkpoint)
                result.best_occ_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
                result.mlflow_run_ids.append(run.info.run_id)

        if cfg.model in ("piece", "both"):
            piece_checkpoint = run_dir / "piece.pt"
            with mlops.training_run(
                mlops.EXPERIMENTS["piece"],
                cfg.name,
                shared_params,
            ) as run:
                history = train_model(
                    csv_path=squares_csv,
                    root=root,
                    label_map={lbl: idx for idx, lbl in enumerate(_PIECE_LABELS_NO_EMPTY)},
                    num_classes=12,
                    max_epochs=cfg.max_epochs,
                    output_path=piece_checkpoint,
                    device=device,
                    model_name=f"Piece[{cfg.name}]",
                    class_names=list(_PIECE_LABELS_NO_EMPTY),  # type: ignore[arg-type]
                    learning_rate=cfg.lr,
                    class_weighted_loss=True,
                    upsample_minority_classes=True,
                    image_column="piece_image_path",
                    patience=cfg.patience,
                    batch_size=cfg.batch_size,
                    on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step),
                )
                mlops.log_artifact(piece_checkpoint)
                result.best_piece_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
                result.mlflow_run_ids.append(run.info.run_id)

    except Exception as exc:
        LOGGER.error(f"Config {cfg.name} failed: {exc}", exc_info=True)
        result.error = str(exc)

    return result


def _print_summary(results: list[RunResult]) -> None:
    """Print a compact result table to stdout."""
    col_name = max(len(r.name) for r in results)
    header = f"{'Config':<{col_name}}  {'Model':<10}  {'Occ val_loss':>12}  {'Piece val_loss':>14}  Status"
    LOGGER.info("=== Hyperparameter search summary ===")
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for r in results:
        occ = f"{r.best_occ_val_loss:.5f}" if r.best_occ_val_loss < float("inf") else "—"
        piece = f"{r.best_piece_val_loss:.5f}" if r.best_piece_val_loss < float("inf") else "—"
        status = f"ERROR: {r.error}" if r.error else "OK"
        LOGGER.info(
            f"{r.name:<{col_name}}  {r.model:<10}  {occ:>12}  {piece:>14}  {status}"
        )


def run_search(
    squares_csv: Path,
    output_dir: Path,
    configs: list[RunConfig],
) -> list[RunResult]:
    """Train each config in *configs* sequentially and return results.

    Args:
        squares_csv: Path to the squares.csv manifest.
        output_dir: Root directory; each config gets its own sub-directory.
        configs: Ordered list of configurations to run.

    Returns:
        List of :class:`RunResult` in the same order as *configs*.
    """
    LOGGER.info(
        f"Starting hyperparameter search configs={len(configs)} output={output_dir}"
    )
    results: list[RunResult] = []
    for i, cfg in enumerate(configs, start=1):
        LOGGER.info(f"--- Config {i}/{len(configs)}: {cfg.name} ---")
        results.append(_train_one(cfg, squares_csv, output_dir))

    _print_summary(results)
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a model for each hyperparameter config in a JSON file."
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
        default=Path("models/search"),
        help="Root output directory; each config gets its own sub-directory (default: models/search/).",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        required=True,
        help="JSON file listing hyperparameter configs (see script docstring for format).",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    configs = _parse_configs(args.configs)
    LOGGER.info(f"Loaded {len(configs)} configs from {args.configs}")
    for cfg in configs:
        LOGGER.info(
            f"  {cfg.name}: lr={cfg.lr} batch_size={cfg.batch_size} "
            f"patience={cfg.patience} max_epochs={cfg.max_epochs} model={cfg.model}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    run_search(args.squares, args.output, configs)


if __name__ == "__main__":
    main()
