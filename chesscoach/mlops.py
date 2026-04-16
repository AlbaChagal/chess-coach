"""Thin MLflow wrappers for the chesscoach training pipeline.

Keeps experiment-tracking concerns out of core training logic.
Scripts import this module instead of calling ``mlflow`` directly.

Usage example::

    with mlops.training_run("chesscoach-occupancy", "run-001", params) as run:
        history = train_model(..., on_epoch=lambda step, m: mlops.log_epoch_metrics(m, step))
        mlops.log_artifact(checkpoint_path)
        mlops.register_checkpoint(checkpoint_path, "OccupancyClassifier")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

LOGGER = logging.getLogger(__name__)

# Canonical experiment names
EXPERIMENTS: dict[str, str] = {
    "occupancy": "chesscoach-occupancy",
    "piece": "chesscoach-piece",
    "transfer": "chesscoach-transfer",
    "benchmark": "chesscoach-benchmark",
    "hp_search": "chesscoach-hp-search",
}


@contextmanager
def training_run(
    experiment: str,
    run_name: str,
    params: dict[str, object],
    *,
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun]:
    """Set the MLflow experiment, start a run, log *params*, and yield.

    Args:
        experiment: Experiment name (use a value from :data:`EXPERIMENTS`).
        run_name: Human-readable name for this run.
        params: Hyperparameters / config to log on the run.
        nested: If ``True``, create a child run inside the current active run.

    Yields:
        The active :class:`mlflow.ActiveRun` context.
    """
    mlflow.set_experiment(experiment)
    LOGGER.debug(
        f"Starting MLflow run experiment={experiment} run_name={run_name} "
        f"nested={nested} params={params}"
    )
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        LOGGER.info(
            f"MLflow run started experiment={experiment} run_name={run_name} "
            f"run_id={run.info.run_id}"
        )
        yield run
        LOGGER.info(
            f"MLflow run finished experiment={experiment} run_id={run.info.run_id}"
        )


def log_epoch_metrics(metrics: dict[str, float], step: int) -> None:
    """Log a dict of scalar metrics at *step*.

    No-op when there is no active MLflow run (e.g. in tests or stub mode).

    Args:
        metrics: Metric name → value mapping.
        step: The epoch index (1-based).
    """
    if mlflow.active_run() is None:
        return
    LOGGER.debug(f"Logging epoch metrics step={step} metrics={metrics}")
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path) -> None:
    """Log *path* as an artifact on the active run.

    No-op when there is no active run.

    Args:
        path: Local file to upload as an artifact.
    """
    if mlflow.active_run() is None:
        return
    LOGGER.debug(f"Logging artifact path={path}")
    mlflow.log_artifact(str(path))


def register_checkpoint(
    checkpoint_path: Path,
    registry_name: str,
) -> str | None:
    """Log *checkpoint_path* and register it in the MLflow Model Registry.

    The file is logged as an artifact and then registered under *registry_name*.

    Args:
        checkpoint_path: Local ``.pt`` file to register.
        registry_name: Registry model name (e.g. ``"OccupancyClassifier"``).

    Returns:
        The registered model version string (``"1"``, ``"2"``, …), or
        ``None`` when the checkpoint could not be registered.

    Raises:
        RuntimeError: If there is no active MLflow run.
    """
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("register_checkpoint() called outside of an active MLflow run.")

    run_id = active.info.run_id
    artifact_name = checkpoint_path.name

    mlflow.log_artifact(str(checkpoint_path))
    model_uri = f"runs:/{run_id}/{artifact_name}"
    try:
        registered = mlflow.register_model(model_uri=model_uri, name=registry_name)
    except MlflowException as exc:
        LOGGER.warning(
            f"Skipping model registry registration for checkpoint={checkpoint_path.name} "
            f"registry={registry_name} run_id={run_id}: {exc}"
        )
        return None

    version = registered.version
    LOGGER.info(
        f"Registered model checkpoint={checkpoint_path.name} "
        f"registry={registry_name} version={version}"
    )
    return str(version)
