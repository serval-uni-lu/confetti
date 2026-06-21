import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from confetti.visualizations._theme import get_theme

_FONT_FAMILY = "sans-serif"


def visualize_cam(weights: np.ndarray, instance_index: int, theme: str = "light") -> None:
    """Visualize the Class Activation Map (CAM) for a single instance.

    This function produces a matplotlib plot showing the normalized
    class activation map for a chosen instance. The weights for all instances
    are first min–max normalized across the dataset, and the selected CAM
    is displayed over time as a line chart with a heatmap strip beneath it.

    Parameters
    ----------
    weights : ndarray of shape (n_instances, timesteps)
        The CAM weights computed for each instance.
    instance_index : int
        Index of the instance whose CAM should be visualized.
    theme : {"light", "dark"}, default="light"
        Color theme for the plot.

    Raises
    ------
    IndexError
        If ``instance_index`` is out of range for ``weights``.
    ValueError
        If ``weights`` is not a 2D array.
    CONFETTIConfigurationError
        If ``theme`` is not a recognized value.

    Note
    -----
    The CAM values are normalized using ``MinMaxScaler`` prior to
    visualization to enable clearer comparison across instances.
    """
    t = get_theme(theme)

    # Normalize weights
    scaler = MinMaxScaler()
    normalized_weights = scaler.fit_transform(weights)

    cam_signal = normalized_weights[instance_index]
    time_axis = np.arange(len(cam_signal))

    # Clear any previous styling
    plt.style.use("default")

    fig, (ax_line, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(13, 5.5),
        gridspec_kw={"height_ratios": [3, 0.6], "hspace": 0.12},
        sharex=True,
    )

    fig.set_facecolor(t["bg"])
    for ax in (ax_line, ax_heat):
        ax.set_facecolor(t["panel"])
        ax.tick_params(colors=t["text_dim"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(t["grid"])
            spine.set_linewidth(0.6)

    color_orig = t["original"]
    color_cf = t["counterfactual"]

    ax_line.plot(time_axis, cam_signal, color=color_orig, linewidth=1.8, alpha=0.95, zorder=3)

    peak_idx = np.argmax(cam_signal)
    ax_line.scatter(
        [peak_idx],
        [cam_signal[peak_idx]],
        color=color_cf,
        s=40,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax_line.annotate(
        f"peak {cam_signal[peak_idx]:.2f}",
        xy=(peak_idx, cam_signal[peak_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        color=color_cf,
        family=_FONT_FAMILY,
    )

    ax_line.set_ylabel("Normalized Weight", fontsize=10, color=t["text_dim"], family=_FONT_FAMILY)
    ax_line.set_ylim(0, 1)
    ax_line.grid(True, linestyle="-", linewidth=0.3, color=t["grid"], alpha=0.6)

    ax_heat.imshow(
        cam_signal[np.newaxis, :],
        aspect="auto",
        cmap=t["heatmap_cmap"],
        extent=[0, len(cam_signal), 0, 1],
        interpolation="bilinear",
    )
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Time Steps", fontsize=10, color=t["text_dim"], family=_FONT_FAMILY)
    ax_heat.grid(False)

    fig.suptitle(
        f"Class Activation Map for Instance {instance_index}",
        y=0.97,
        fontsize=14,
        fontweight="semibold",
        color=t["text"],
        family=_FONT_FAMILY,
    )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10)
    plt.show()
