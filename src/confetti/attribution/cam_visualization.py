import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def visualize_cam(weights: np.ndarray, instance_index: int) -> None:
    """
    Visualize the Class Activation Map (CAM) for a specific instance using pure matplotlib,
    with a clean white-background design.

    Parameters:
        weights (np.ndarray): CAM weights for multiple instances.
        instance_index (int): Index of the instance to visualize.
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
