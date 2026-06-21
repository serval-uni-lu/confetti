"""Tests for the lazy-import helpers in utils/_compat.py."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from confetti.errors import CONFETTIConfigurationError
from confetti.utils._compat import require_keras, require_torch, get_keras, get_torch


# ---------------------------------------------------------------------------
# require_keras
# ---------------------------------------------------------------------------


def test_require_keras_missing_raises():
    with patch.dict("sys.modules", {"keras": None}):
        with pytest.raises(CONFETTIConfigurationError) as exc_info:
            require_keras("unit test")

        err = exc_info.value
        assert "unit test" in err.message
        assert "pip install confetti-ts[keras]" in err.hint


def test_require_keras_present_returns_module():
    try:
        import keras  # noqa: F401
    except ImportError:
        pytest.skip("keras not installed")

    mod = require_keras("unit test")
    assert hasattr(mod, "__version__")


# ---------------------------------------------------------------------------
# require_torch
# ---------------------------------------------------------------------------


def test_require_torch_missing_raises():
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(CONFETTIConfigurationError) as exc_info:
            require_torch("unit test")

        err = exc_info.value
        assert "unit test" in err.message
        assert "pip install confetti-ts[torch]" in err.hint


def test_require_torch_present_returns_module():
    torch = pytest.importorskip("torch")
    mod = require_torch("unit test")
    assert hasattr(mod, "tensor")


# ---------------------------------------------------------------------------
# get_keras / get_torch (None-returning variants)
# ---------------------------------------------------------------------------


def test_get_keras_missing_returns_none():
    with patch.dict("sys.modules", {"keras": None}):
        assert get_keras() is None


def test_get_torch_missing_returns_none():
    with patch.dict("sys.modules", {"torch": None}):
        assert get_torch() is None


def test_get_torch_present_returns_module():
    torch = pytest.importorskip("torch")
    mod = get_torch()
    assert mod is not None
    assert hasattr(mod, "tensor")
