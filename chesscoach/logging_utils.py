"""Shared logging helpers for CLI entry points."""

from __future__ import annotations

import argparse
import logging
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VAR = "CHESSCOACH_LOG_LEVEL"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add standard logging CLI arguments to a parser."""
    parser.add_argument(
        "--log-level",
        default=os.environ.get(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL),
        help=(
            "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
            f"Default: {DEFAULT_LOG_LEVEL} or ${LOG_LEVEL_ENV_VAR}."
        ),
    )


def configure_logging(level: str) -> int:
    """Configure root logging and return the resolved numeric level."""
    numeric_level = _parse_log_level(level)
    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    return numeric_level


def _parse_log_level(level: str) -> int:
    """Resolve a user-provided logging level string."""
    normalized = level.upper()
    numeric_level = getattr(logging, normalized, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")
    return numeric_level
