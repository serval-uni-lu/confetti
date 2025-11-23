import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def visualize_cam(weights: np.ndarray, instance_index: int) -> None:
    """Visualize the Class Activation Map (CAM) for a single instance.

    This function produces a simple matplotlib plot showing the normalized
    class activation map for a chosen instance. The weights for all instances
    are first min–max normalized across the dataset, and the selected CAM
    is displayed over time with a clean white-background design.

    Parameters
    ----------
    weights : ndarray of shape (n_instances, timesteps)
        The CAM weights computed for each instance.
    instance_index : int
        Index of the instance whose CAM should be visualized.

    Raises
    ------
    IndexError
        If ``instance_index`` is out of range for ``weights``.
    ValueError
        If ``weights`` is not a 2D array.

    Note
    -----
    The CAM values are normalized using ``MinMaxScaler`` prior to
    visualization to enable clearer comparison across instances.
    """

    # Normalize weights
    scaler = MinMaxScaler()
    normalized_weights = scaler.fit_transform(weights)

    cam_signal = normalized_weights[instance_index]

    # Clear any previous styling
    plt.style.use("default")

    plt.figure(figsize=(10, 6), facecolor="white")

    plt.plot(
        cam_signal,
        color="#018575",
        linewidth=2,
        alpha=0.9,
    )

    # White background and clean grid
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#DDDDDD", alpha=0.8)

    # Titles and labels
    plt.title(f"Class Activation Map for Instance {instance_index}", fontsize=16)
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Normalized Weight", fontsize=14)
    plt.ylim(0, 1)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
