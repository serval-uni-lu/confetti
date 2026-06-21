"""Model adapters that normalise different framework interfaces to a common predict API."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from confetti.utils._compat import require_torch


@runtime_checkable
class Predictor(Protocol):
    """Any object whose ``predict`` method maps numpy arrays to numpy arrays."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class TorchModelAdapter:
    """Wrap a :class:`torch.nn.Module` so it exposes a ``.predict()`` interface.

    The adapter handles:
    * eval mode and ``torch.no_grad``
    * numpy ↔ tensor conversion
    * dimension reordering from CONFETTI's channels-last ``(N, T, C)``
      to PyTorch's channels-first ``(N, C, T)`` convention

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model.
    device : str or None
        Device to run inference on.  Defaults to ``"cuda"`` when available,
        otherwise ``"cpu"``.
    channels_last : bool
        If ``True`` (default), input arrays use CONFETTI's ``(N, T, C)``
        layout and are permuted to ``(N, C, T)`` before calling the model.
        Set to ``False`` if the model already expects ``(N, T, C)``.
    """

    def __init__(self, model, device: str | None = None, channels_last: bool = True):
        torch = require_torch("TorchModelAdapter")
        self._model = model
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._channels_last = channels_last

    @property
    def torch_model(self):
        """Access the underlying :class:`torch.nn.Module` (needed for CAM)."""
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and return class probabilities as a numpy array.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, timesteps, channels)``
            Input time series in CONFETTI's channels-last layout.

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
        """
        import torch

        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32, device=self._device)
            if self._channels_last:
                tensor = tensor.permute(0, 2, 1)
            output = self._model(tensor)
            return output.cpu().numpy()
