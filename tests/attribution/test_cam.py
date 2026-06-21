"""Tests for Class Activation Map computation (attribution/cam.py)."""

from __future__ import annotations

import numpy as np
import pytest

from confetti.errors import CONFETTIConfigurationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES, TIMESTEPS, CHANNELS, N_CLASSES = 4, 20, 3, 2


@pytest.fixture
def sample_input():
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, TIMESTEPS, CHANNELS)).astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch CAM tests
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from confetti.adapters import TorchModelAdapter  # noqa: E402
from confetti.attribution.cam import cam  # noqa: E402


class _TorchCAMModel(nn.Module):
    """Minimal Conv1d → Linear model suitable for CAM."""

    def __init__(self, n_channels=CHANNELS, n_filters=8, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_filters, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class _TorchNoConvModel(nn.Module):
    """Model without any Conv1d layer."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CHANNELS * TIMESTEPS, N_CLASSES)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


class _TorchNoLinearModel(nn.Module):
    """Model with Conv1d but no Linear layer."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(CHANNELS, N_CLASSES, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.conv(x)).squeeze(-1)


@pytest.fixture
def torch_cam_model():
    return _TorchCAMModel()


# -- output shape -----------------------------------------------------------


def test_cam_torch_output_shape(torch_cam_model, sample_input):
    result = cam(torch_cam_model, sample_input)
    assert result.shape == (N_SAMPLES, TIMESTEPS)


def test_cam_torch_adapter_output_shape(torch_cam_model, sample_input):
    adapter = TorchModelAdapter(torch_cam_model, device="cpu")
    result = cam(adapter, sample_input)
    assert result.shape == (N_SAMPLES, TIMESTEPS)


# -- error paths ------------------------------------------------------------


def test_cam_torch_no_conv_raises(sample_input):
    model = _TorchNoConvModel()
    with pytest.raises(CONFETTIConfigurationError, match="Conv1d"):
        cam(model, sample_input)


def test_cam_torch_no_linear_raises(sample_input):
    model = _TorchNoLinearModel()
    with pytest.raises(CONFETTIConfigurationError, match="Linear"):
        cam(model, sample_input)


# -- determinism ------------------------------------------------------------


def test_cam_torch_deterministic(torch_cam_model, sample_input):
    r1 = cam(torch_cam_model, sample_input)
    r2 = cam(torch_cam_model, sample_input)
    np.testing.assert_array_equal(r1, r2)


# -- single sample ----------------------------------------------------------


def test_cam_torch_single_sample(torch_cam_model):
    rng = np.random.default_rng(99)
    x = rng.standard_normal((1, TIMESTEPS, CHANNELS)).astype(np.float32)
    result = cam(torch_cam_model, x)
    assert result.shape == (1, TIMESTEPS)
