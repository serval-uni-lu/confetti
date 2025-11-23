from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from confetti.errors import (
    CONFETTIConfigurationError,
    CONFETTIDataTypeError,
)
from confetti.structs import Counterfactual


def _normalize_series(series: np.ndarray | Counterfactual, param_name: str) -> np.ndarray:
    """
    Normalize a time series to shape ``(timesteps, channels)``.

    This helper accepts the following input shapes:

    - ``(timesteps, channels)`` — returned unchanged
    - ``(1, timesteps, channels)`` — squeezed to ``(timesteps, channels)``
    - ``Counterfactual`` — its internal array is used

    Parameters
    ----------
    series : ndarray or Counterfactual
        Input time series.
    param_name : str
        Name of the parameter (used for error reporting).

    Returns
    -------
    ndarray of shape (timesteps, channels)
        The normalized time series.

    Raises
    ------
    CONFETTIDataTypeError
        If the input is not a supported type or shape.
    """
    if not isinstance(series, (np.ndarray, Counterfactual)):
        raise CONFETTIDataTypeError(
            message=f"{param_name} must be a numpy ndarray or Counterfactual object.",
            param=param_name,
            hint="Ensure you pass a numpy array with shape (timesteps, channels) or (1, timesteps, channels) or a Counterfactual object.",
        )
    if isinstance(series, Counterfactual):
        series = series.counterfactual

    if series.ndim == 2:
        # (timesteps, channels)
        return series

    if series.ndim == 3 and series.shape[0] == 1:
        # (1, timesteps, channels) -> (timesteps, channels)
        return series[0]

    raise CONFETTIDataTypeError(
        message=(
            f"{param_name} must have shape (timesteps, channels) or "
            f"(1, timesteps, channels), but got shape {series.shape}."
        ),
        param=param_name,
        hint="Use series.reshape(timesteps, channels) or series[np.newaxis, ...] if needed.",
    )


def _select_channels(
    series: np.ndarray,
    channels: Optional[Sequence[int]],
) -> Tuple[np.ndarray, Sequence[int]]:
    """
    Select specific channels from a multivariate time series.

    Parameters
    ----------
    series : ndarray of shape (timesteps, channels)
        Multivariate time series.
    channels : sequence of int or None
        Indices of channels to keep. If ``None``, all channels are used.

    Returns
    -------
    tuple
        ``(series_subset, selected_indices)`` where:

        - ``series_subset`` is the sliced array with selected channels
        - ``selected_indices`` is the list of channel indices actually used

    Raises
    ------
    CONFETTIConfigurationError
        If a channel index is out of bounds.
    """

    timesteps, n_channels = series.shape

    if channels is not None:
        selected_idx = list(channels)
        for idx in selected_idx:
            if idx < 0 or idx >= n_channels:
                raise CONFETTIConfigurationError(
                    message=(
                        f"Channel index {idx} is out of bounds for a series with "
                        f"{n_channels} channels."
                    ),
                    param="channels",
                    hint="Ensure channel indices are 0-based and strictly less than the number of channels.",
                )
        return series[:, selected_idx], selected_idx

    selected_idx = list(range(n_channels))
    return series[:, selected_idx], selected_idx



def plot_time_series(
    series: np.ndarray | Counterfactual,
    channels: Optional[Sequence[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot a multivariate time series.

    Parameters
    ----------
    series : ndarray or Counterfactual
        Time series with shape ``(timesteps, channels)`` or
        ``(1, timesteps, channels)``.
    channels : sequence of int, optional
        Channel indices to plot. If ``None``, all channels are plotted.
    figsize : tuple of float, optional
        Figure size passed to Matplotlib. If ``None``, determined automatically.
    title : str, optional
        Title of the figure.

    Raises
    ------
    CONFETTIDataTypeError
        If ``series`` has an unsupported shape.
    CONFETTIConfigurationError
        If ``channels`` contains invalid indices.
    """
    series_2d = _normalize_series(series, param_name="series")
    series_selected, selected_idx = _select_channels(series_2d, channels)

    timesteps, n_channels = series_selected.shape

    if figsize is None:
        figsize = (12.0, 3.0 * n_channels)

    plt.style.use("default")
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    if n_channels == 1:
        axes = [axes]

    time_axis = np.arange(timesteps)

    for i, ax in enumerate(axes):
        chan_idx = selected_idx[i]
        ax.plot(time_axis, series_selected[:, i], linewidth=2.0, label=f"Channel {chan_idx}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")

    if title is not None:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.show()


def plot_counterfactual(
    original: np.ndarray,
    counterfactual: Counterfactual,
    channels: Optional[Sequence[int]] = None,
    cam_weights: Optional[np.ndarray] = None,
    cam_mode: str = "none",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot an original multivariate time series together with its counterfactual.

    The counterfactual is assumed to differ from the original over a contiguous
    subsequence, which is highlighted in the visualization. Optional CAM
    information can be overlaid either as a line or as a heatmap.

    Parameters
    ----------
    original : ndarray
        Original time series with shape ``(timesteps, channels)`` or
        ``(1, timesteps, channels)``.
    counterfactual : Counterfactual
        Counterfactual object whose internal ``counterfactual`` array has the
        same shape as ``original``.
    channels : sequence of int, optional
        Channel indices to plot. If ``None``, all channels are shown.
    cam_weights : ndarray of shape (timesteps,), optional
        CAM weights to overlay. Used when ``cam_mode`` is ``"line"`` or
        ``"heatmap"``.
    cam_mode : {"none", "line", "heatmap"}, default="none"
        Type of CAM visualization to include:

        - ``"none"``: no CAM overlay
        - ``"line"``: CAM shown as a line in each subplot
        - ``"heatmap"``: CAM shown as a separate heatmap subplot
    figsize : tuple of float, optional
        Figure size passed to Matplotlib. If ``None``, chosen automatically.
    title : str, optional
        Optional title for the figure.

    Raises
    ------
    CONFETTIDataTypeError
        If the time series arrays have incompatible shapes or invalid types.
    CONFETTIConfigurationError
        If ``channels`` are invalid or if CAM settings are inconsistent.
    """
    original_2d = _normalize_series(original, param_name="original")

    if not isinstance(counterfactual, Counterfactual):
        raise CONFETTIDataTypeError(
            message="counterfactual must be a Counterfactual object.",
            param="counterfactual",
            hint="Use the Counterfactual class from confetti.explainer.counterfactuals.",
        )

    cf_array = _normalize_series(counterfactual.counterfactual, param_name="counterfactual.counterfactual")

    if original_2d.shape != cf_array.shape:
        raise CONFETTIDataTypeError(
            message=(
                "original and counterfactual must have the same shape "
                f"after normalization, but got {original_2d.shape} and {cf_array.shape}."
            ),
            param="original / counterfactual.counterfactual",
            hint="Ensure both arrays correspond to the same instance and have identical shapes.",
        )

    timesteps, _ = original_2d.shape
    series_original, selected_idx = _select_channels(original_2d, channels)
    series_cf, _ = _select_channels(cf_array, channels)

    n_channels = series_original.shape[1]
    time_axis = np.arange(timesteps)

    if cam_mode not in {"none", "line", "heatmap"}:
        raise CONFETTIConfigurationError(
            message=f"Invalid cam_mode '{cam_mode}'. Expected 'none', 'line', or 'heatmap'.",
            param="cam_mode",
            hint="Use one of: 'none', 'line', 'heatmap'.",
        )

    if cam_mode in {"line", "heatmap"} and cam_weights is not None:
        if not isinstance(cam_weights, np.ndarray) or cam_weights.ndim != 1:
            raise CONFETTIDataTypeError(
                message="cam_weights must be a 1D numpy array.",
                param="cam_weights",
                hint="Provide CAM weights with shape (timesteps,).",
            )
        if cam_weights.shape[0] != timesteps:
            raise CONFETTIConfigurationError(
                message=(
                    f"cam_weights length {cam_weights.shape[0]} does not match "
                    f"time dimension {timesteps}."
                ),
                param="cam_weights",
                hint="Ensure CAM weights are computed for the same number of timesteps as the series.",
            )

    # Decide layout for CAM heatmap variant
    if cam_mode == "heatmap" and cam_weights is not None:
        n_rows = n_channels + 1
        height_ratios = [3] * n_channels + [0.75]
    else:
        n_rows = n_channels
        height_ratios = None

    if figsize is None:
        base_height = 3.0
        extra = 1.5 if cam_mode == "heatmap" and cam_weights is not None else 0.0
        figsize = (12.0, base_height * n_channels + extra)

    plt.style.use("default")
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios} if height_ratios is not None else None,
    )

    if n_rows == 1:
        axes = [axes]

    ts_axes = axes[:n_channels]
    cam_axis = axes[-1] if n_rows > n_channels else None

    # Normalize CAM for plotting (when used)
    cam_normalized = None
    if cam_weights is not None and cam_mode in {"line", "heatmap"}:
        w_min, w_max = float(cam_weights.min()), float(cam_weights.max())
        if w_max > w_min:
            cam_normalized = (cam_weights - w_min) / (w_max - w_min)
        else:
            cam_normalized = np.zeros_like(cam_weights)

    for i, ax in enumerate(ts_axes):
        chan_idx = selected_idx[i]
        orig_dim = series_original[:, i]
        cf_dim = series_cf[:, i]

        ax.plot(time_axis, orig_dim, linewidth=2.0, label="Original", color="#64cdc0")

        diffs = np.where(orig_dim != cf_dim)[0]
        if diffs.size > 0:
            start_diff, end_diff = diffs[0], diffs[-1]

            ax.plot(
                time_axis[start_diff : end_diff + 1],
                cf_dim[start_diff : end_diff + 1],
                linewidth=2.0,
                linestyle="--",
                color="#ff595a",
                label="Counterfactual",
            )

            if start_diff > 0:
                ax.plot(
                    [time_axis[start_diff - 1], time_axis[start_diff]],
                    [orig_dim[start_diff - 1], cf_dim[start_diff]],
                    linewidth=2.0,
                    color="#ff595a",
                )
            if end_diff < len(orig_dim) - 1:
                ax.plot(
                    [time_axis[end_diff], time_axis[end_diff + 1]],
                    [cf_dim[end_diff], orig_dim[end_diff + 1]],
                    linewidth=2.0,
                    color="#ff595a",
                )

        if cam_mode == "line" and cam_normalized is not None:
            scaled_cam = (cam_normalized * (orig_dim.max() - orig_dim.min())) + orig_dim.min()
            ax.plot(
                time_axis,
                scaled_cam,
                linewidth=1.5,
                linestyle=":",
                color="#ffa500",
                label="CAM",
            )

        ax.set_ylabel("Value")
        ax.set_title(f"Channel {chan_idx}")
        ax.grid(True, linestyle=":", alpha=0.4)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")

    ts_axes[-1].set_xlabel("Time")

    if cam_mode == "heatmap" and cam_normalized is not None and cam_axis is not None:
        cam_axis.imshow(
            cam_normalized[np.newaxis, :],
            aspect="auto",
            cmap="YlOrRd",
            extent=[0, timesteps, 0, 1],
        )
        cam_axis.set_yticks([])
        cam_axis.set_ylabel("CAM")
        cam_axis.grid(False)

    if title is not None:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.show()
