"""Tests for the model adapter layer (adapters.py)."""

from __future__ import annotations

import numpy as np
import pytest

from confetti.adapters import Predictor


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------


class _PlainPredictor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X


class _NotAPredictor:
    def score(self, X: np.ndarray) -> float:
        return 0.0


def test_predictor_protocol_satisfied():
    assert isinstance(_PlainPredictor(), Predictor)


def test_predictor_protocol_not_satisfied():
    assert not isinstance(_NotAPredictor(), Predictor)


# ---------------------------------------------------------------------------
# TorchModelAdapter
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from confetti.adapters import TorchModelAdapter  # noqa: E402


class _TinyModel(nn.Module):
    """Conv1d(C_in, 8) → Linear(8, n_classes). Expects (N, C, T)."""

    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@pytest.fixture
def tiny_model():
    return _TinyModel(n_channels=3, n_classes=2)


@pytest.fixture
def sample_input():
    """(N=4, T=10, C=3) channels-last input."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 10, 3)).astype(np.float32)


# -- __init__ ---------------------------------------------------------------


def test_init_defaults_to_cpu(tiny_model):
    adapter = TorchModelAdapter(tiny_model)
    assert adapter._device in ("cpu", "cuda")


def test_init_explicit_device(tiny_model):
    adapter = TorchModelAdapter(tiny_model, device="cpu")
    assert adapter._device == "cpu"


def test_init_sets_eval_mode(tiny_model):
    tiny_model.train()
    adapter = TorchModelAdapter(tiny_model)
    assert not adapter._model.training


def test_init_stores_channels_last_flag(tiny_model):
    adapter_default = TorchModelAdapter(tiny_model)
    assert adapter_default._channels_last is True

    adapter_false = TorchModelAdapter(tiny_model, channels_last=False)
    assert adapter_false._channels_last is False


# -- torch_model property ---------------------------------------------------


def test_torch_model_property(tiny_model):
    adapter = TorchModelAdapter(tiny_model)
    assert adapter.torch_model is tiny_model


# -- predict -----------------------------------------------------------------


def test_predict_output_shape(tiny_model, sample_input):
    adapter = TorchModelAdapter(tiny_model, device="cpu")
    out = adapter.predict(sample_input)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 2)


def test_predict_channels_last_permutation(sample_input):
    """Verify channels-last input is permuted to channels-first before model."""
    received = {}

    class _RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, x):
            received["shape"] = tuple(x.shape)
            return self.fc(x.mean(dim=-1))

    model = _RecordingModel()
    adapter = TorchModelAdapter(model, device="cpu", channels_last=True)
    adapter.predict(sample_input)
    # channels-last (4, 10, 3) → channels-first (4, 3, 10)
    assert received["shape"] == (4, 3, 10)


def test_predict_channels_first_skips_permutation(sample_input):
    """When channels_last=False, input shape is forwarded as-is."""
    received = {}

    class _RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, x):
            received["shape"] = tuple(x.shape)
            return self.fc(x.mean(dim=1))

    model = _RecordingModel()
    adapter = TorchModelAdapter(model, device="cpu", channels_last=False)
    adapter.predict(sample_input)
    assert received["shape"] == (4, 10, 3)


def test_predict_returns_numpy(tiny_model, sample_input):
    adapter = TorchModelAdapter(tiny_model, device="cpu")
    out = adapter.predict(sample_input)
    assert isinstance(out, np.ndarray)
    assert out.dtype in (np.float32, np.float64)


def test_predict_deterministic(tiny_model, sample_input):
    adapter = TorchModelAdapter(tiny_model, device="cpu")
    out1 = adapter.predict(sample_input)
    out2 = adapter.predict(sample_input)
    np.testing.assert_array_equal(out1, out2)


# -- Predictor protocol conformance -----------------------------------------


def test_adapter_satisfies_predictor_protocol(tiny_model):
    adapter = TorchModelAdapter(tiny_model, device="cpu")
    assert isinstance(adapter, Predictor)
