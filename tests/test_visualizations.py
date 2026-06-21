from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from confetti.attribution import visualize_cam
from confetti.errors import CONFETTIConfigurationError
from confetti.structs import Counterfactual
from confetti.visualizations import plot_counterfactual, plot_time_series


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def sample_series():
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 2))


@pytest.fixture()
def sample_counterfactual(sample_series):
    cf_data = sample_series.copy()
    cf_data[10:20] += 1.0
    return Counterfactual(counterfactual=cf_data, label=1)


@pytest.fixture()
def sample_cam_weights():
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(size=(3, 50)))


class TestThemeValidation:
    def test_plot_time_series_rejects_invalid_theme(self, sample_series):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            plot_time_series(series=sample_series, theme="neon")

    def test_plot_counterfactual_rejects_invalid_theme(self, sample_series, sample_counterfactual):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            plot_counterfactual(original=sample_series, counterfactual=sample_counterfactual, theme="neon")

    def test_visualize_cam_rejects_invalid_theme(self, sample_cam_weights):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            visualize_cam(weights=sample_cam_weights, instance_index=0, theme="neon")

    def test_plot_time_series_rejects_empty_theme(self, sample_series):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            plot_time_series(series=sample_series, theme="")

    def test_plot_counterfactual_rejects_empty_theme(self, sample_series, sample_counterfactual):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            plot_counterfactual(original=sample_series, counterfactual=sample_counterfactual, theme="")

    def test_visualize_cam_rejects_empty_theme(self, sample_cam_weights):
        with pytest.raises(CONFETTIConfigurationError, match="Invalid theme"):
            visualize_cam(weights=sample_cam_weights, instance_index=0, theme="")

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_plot_time_series_accepts_valid_themes(self, sample_series, theme):
        plot_time_series(series=sample_series, theme=theme)

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_plot_counterfactual_accepts_valid_themes(self, sample_series, sample_counterfactual, theme):
        plot_counterfactual(original=sample_series, counterfactual=sample_counterfactual, theme=theme)

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_visualize_cam_accepts_valid_themes(self, sample_cam_weights, theme):
        visualize_cam(weights=sample_cam_weights, instance_index=0, theme=theme)

    def test_theme_error_includes_param_name(self, sample_series):
        with pytest.raises(CONFETTIConfigurationError) as exc_info:
            plot_time_series(series=sample_series, theme="bad")
        assert exc_info.value.param == "theme"

    def test_theme_error_includes_hint(self, sample_series):
        with pytest.raises(CONFETTIConfigurationError) as exc_info:
            plot_time_series(series=sample_series, theme="bad")
        assert "light" in exc_info.value.hint
        assert "dark" in exc_info.value.hint
