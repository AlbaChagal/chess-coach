"""Tests for PyTorch device selection."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from chesscoach import torch_utils


def test_select_device_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch_utils.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert torch_utils.select_device() == torch.device("mps")


def test_select_device_uses_cuda_when_mps_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch_utils.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    assert torch_utils.select_device() == torch.device("cuda")


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch_utils.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    assert torch_utils.select_device() == torch.device("cpu")
