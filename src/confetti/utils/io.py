import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from confetti.errors import CONFETTIConfigurationError


def convert_string_to_array(data_string: str, timesteps: int, channels: int) -> np.ndarray:
    """Convert a serialized array string into a 2D NumPy array.

    This function reconstructs a flattened, space-separated string
    representation of an array into a 2D array of shape
    ``(timesteps, channels)``. Brackets and line breaks are removed
    automatically.

    Parameters
    ----------
    data_string : str
        The flattened array stored as a whitespace-separated string.
    timesteps : int
        Number of expected time steps in the reconstructed array.
    channels : int
        Number of expected channels in the reconstructed array.

    Returns
    -------
    ndarray of shape (timesteps, channels)
        The reconstructed numeric array.

    Raises
    ------
    CONFETTIConfigurationError
        If the number of parsed elements does not match
        ``timesteps * channels``.
    """
    cleaned = data_string.replace("[", "").replace("]", "").replace("\n", " ")
    try:
        values = np.fromstring(cleaned, sep=" ")
    except ValueError as exc:
        raise CONFETTIConfigurationError(
            f"Failed to parse numeric values from input: {data_string}",
            param="data_string",
            hint="Ensure the string contains only numeric values separated by spaces.",
            source="convert_string_to_array",
        ) from exc

    expected_size = timesteps * channels
    if values.size != expected_size:
        raise CONFETTIConfigurationError(
            message=(
                f"Data does not match expected size ({timesteps}, {channels}). "
                f"Found {values.size} elements."
            ),
            config={"data_string": data_string},
            param="data_string",
            hint="Ensure the flattened string contains the correct number of whitespace-separated values.",
            source="convert_string_to_array",
        )

    return values.reshape(timesteps, channels)


def save_multivariate_ts_as_csv(file_path: str | Path, x: np.ndarray, y: np.ndarray) -> None:
    """Save a multivariate time-series dataset to CSV in long format.

    The time-series array ``x`` is converted into a long-format table where
    each row corresponds to one sample–time-step pair. The function also
    writes a companion ``*_labels.csv`` file storing a single label per
    sample.

    Parameters
    ----------
    file_path : str or Path
        Destination path for the main CSV file.
    x : ndarray of shape (n_samples, n_time_steps, n_features)
        The multivariate time series data.
    y : ndarray of shape (n_samples,)
        The label associated with each sample.

    Note
    -----
    Two files are written:

    - ``<file>.csv``: long-format feature table
    - ``<file>_labels.csv``: labels indexed by ``sample_id``
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
    """Load a multivariate time-series dataset saved with ``save_multivariate_ts_as_csv``.

    This function reconstructs the original ``X`` array by reshaping the
    long-format CSV and retrieves the sample-level labels from the companion
    ``*_labels.csv`` file.

    Parameters
    ----------
    file_path : str or Path
        Path to the main feature CSV file.

    Returns
    -------
    X : ndarray of shape (n_samples, n_time_steps, n_features)
        The reconstructed multivariate time series.
    y : ndarray of shape (n_samples,)
        The label associated with each sample.

    Note
    -----
    The function expects two files in the same directory:

    - ``<file>.csv``
    - ``<file>_labels.csv``
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
    """Convert a NumPy array into a single-line space-separated string.

    Useful for serializing arrays into CSV files where nested structures
    are not supported.

    Parameters
    ----------
    arr : ndarray
        The array to be flattened and serialized.

    Returns
    -------
    str
        A flattened, space-separated representation of the array.
    """
    return " ".join(map(str, arr.flatten()))
