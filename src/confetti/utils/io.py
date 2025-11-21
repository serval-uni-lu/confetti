import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def convert_string_to_array(data_string: str, timesteps: int, channels: int) -> np.ndarray:
    """
    Convert a serialized array string back into a 2D NumPy array (timesteps, channels).

    Parameters
    ----------
    data_string : str
        String representation of the array, typically flattened and bracketed.
    timesteps : int
        Expected number of time steps.
    channels : int
        Expected number of channels.

    Returns
    -------
    np.ndarray
        Array of shape (timesteps, channels).

    Raises
    ------
    ValueError
        If the number of elements does not match the expected size.
    """
    cleaned = data_string.replace("[", "").replace("]", "").replace("\n", "")
    values = np.fromstring(cleaned, sep=" ")

    expected_size = timesteps * channels
    if values.size != expected_size:
        raise ValueError(
            f"Data does not match expected size ({timesteps}, {channels}). "
            f"Found {values.size} elements."
        )

    return values.reshape(timesteps, channels)


def save_multivariate_ts_as_csv(file_path: str | Path, x: np.ndarray, y: np.ndarray) -> None:
    """
    Save a multivariate time series dataset to CSV in long format. The main CSV stores features per sample and time step. A companion CSV stores one label per sample.

    Parameters
    ----------
    file_path : str or Path
        Destination path for the main CSV file.
    x : np.ndarray
        Array of shape (n_samples, n_time_steps, n_features).
    y : np.ndarray
        Array of shape (n_samples,) containing labels.
    """
    file_path = Path(file_path)
    n_samples, n_time_steps, n_features = x.shape

    flattened = x.reshape(n_samples * n_time_steps, n_features)
    sample_ids = np.repeat(np.arange(n_samples), n_time_steps)
    time_steps = np.tile(np.arange(n_time_steps), n_samples)

    df = pd.DataFrame(flattened, columns=[f"feature_{i}" for i in range(n_features)])
    df.insert(0, "sample_id", sample_ids)
    df.insert(1, "time_step", time_steps)

    label_df = pd.DataFrame({"sample_id": np.arange(n_samples), "label": y})

    df.to_csv(file_path, index=False)
    label_df.to_csv(file_path.with_name(file_path.stem + "_labels.csv"), index=False)


def load_multivariate_ts_from_csv(
        file_path: str | Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a multivariate time series dataset saved using ``save_multivariate_ts_as_csv``.

    Parameters
    ----------
    file_path : str or Path
        Path to the main CSV file.

    Returns
    -------
    tuple
        ``X``: np.ndarray of shape (n_samples, n_time_steps, n_features)
        ``y`` : np.ndarray of shape (n_samples,)
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    label_df = pd.read_csv(file_path.with_name(file_path.stem + "_labels.csv"))

    n_samples = label_df.shape[0]
    n_time_steps = df["time_step"].nunique()
    n_features = df.shape[1] - 2

    feature_values = df.drop(columns=["sample_id", "time_step"]).values
    X = feature_values.reshape(n_samples, n_time_steps, n_features)
    y = label_df["label"].values

    return X, y


def array_to_string(arr: np.ndarray) -> str:
    """
    Convert a NumPy array into a single-line space-separated string.

    Parameters
    ----------
    arr : np.ndarray
        Array to convert.

    Returns
    -------
    str
        Flattened string representation.
    """
    return " ".join(map(str, arr.flatten()))
