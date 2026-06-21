"""Lazy import helpers for optional DL framework dependencies."""

from __future__ import annotations

from confetti.errors import CONFETTIConfigurationError


def get_keras():
    """Return the ``keras`` module, or ``None`` if not installed."""
    try:
        import keras

        return keras
    except ImportError:
        return None


def get_torch():
    """Return the ``torch`` module, or ``None`` if not installed."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def require_keras(operation: str):
    """Return the ``keras`` module or raise with an install hint.

    Parameters
    ----------
    operation : str
        Human-readable description of what needs Keras (used in the error
        message).
    """
    keras = get_keras()
    if keras is None:
        raise CONFETTIConfigurationError(
            message=f"Keras is required for {operation} but is not installed.",
            hint="Install with: pip install confetti-ts[keras]",
        )
    return keras


def require_torch(operation: str):
    """Return the ``torch`` module or raise with an install hint.

    Parameters
    ----------
    operation : str
        Human-readable description of what needs PyTorch (used in the error
        message).
    """
    torch = get_torch()
    if torch is None:
        raise CONFETTIConfigurationError(
            message=f"PyTorch is required for {operation} but is not installed.",
            hint="Install with: pip install confetti-ts[torch]",
        )
    return torch


def is_torch_module(model) -> bool:
    """Check whether *model* is a ``torch.nn.Module`` without requiring torch."""
    try:
        import torch.nn as nn

        return isinstance(model, nn.Module)
    except ImportError:
        return False
