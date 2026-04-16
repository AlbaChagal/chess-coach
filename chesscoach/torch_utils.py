"""Shared PyTorch helpers."""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Return the best available torch device for the current machine."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
