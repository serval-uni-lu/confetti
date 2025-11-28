import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from confetti.utils.io import (
    convert_string_to_array,
    save_multivariate_ts_as_csv,
    load_multivariate_ts_from_csv,
    array_to_string,
)
from confetti.errors import CONFETTIConfigurationError


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def small_array() -> np.ndarray:
    return np.arange(6).reshape(3, 2)


@pytest.fixture
def multivariate_ts() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(24).reshape(4, 3, 2)
    y = np.array([1, 0, 1, 0])
    return X, y


@pytest.fixture
def tmp_csv_paths(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "ts.csv"
    labels = tmp_path / "ts_labels.csv"
    return main, labels


@pytest.mark.parametrize(
    "data_string,timesteps,channels,expected",
    [
        ("1 2 3 4", 2, 2, np.array([[1, 2], [3, 4]])),
        (" 5  6 7  8  ", 2, 2, np.array([[5, 6], [7, 8]])),
        ("9\n10\n11\n12", 2, 2, np.array([[9, 10], [11, 12]])),
    ],
)
def test_convert_string_to_array_valid(data_string, timesteps, channels, expected) -> None:
    """Valid strings reconstruct into the correct array."""
    arr = convert_string_to_array(data_string, timesteps, channels)
    assert np.array_equal(arr, expected)


@pytest.mark.parametrize(
    "data_string,timesteps,channels,found",
    [
        ("1 2 3", 2, 2, 3),   # too few values
        ("", 1, 1, 0),        # empty -> size 0
    ]
)
def test_convert_string_to_array_size_mismatch(data_string, timesteps, channels, found):
    with pytest.raises(CONFETTIConfigurationError) as exc_info:
        convert_string_to_array(data_string, timesteps, channels)

    err = exc_info.value

    assert err.message.startswith("Data does not match expected size")
    assert f"({timesteps}, {channels})" in err.message
    assert f"{found}" in err.message

def test_convert_string_to_array_parse_error():
    bad_string = "a b c"

    with pytest.raises(CONFETTIConfigurationError) as exc_info:
        convert_string_to_array(bad_string, 1, 3)

    err = exc_info.value
    assert err.message.startswith("Failed to parse numeric values")
    assert "a b c" in err.message



def test_array_to_string_roundtrip(small_array: np.ndarray) -> None:
    """Serialization and reconstruction must be consistent."""
    s = array_to_string(small_array)
    reconstructed = convert_string_to_array(s, timesteps=3, channels=2)
    assert np.array_equal(reconstructed, small_array)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([]),
        np.array([[1]]),
        np.arange(12).reshape(2, 2, 3),
    ],
)
def test_array_to_string_edge_cases(arr: np.ndarray) -> None:
    """Edge arrays must serialize without crashing."""
    s = array_to_string(arr)
    assert isinstance(s, str)
    assert "\n" not in s


def test_save_and_load_roundtrip(multivariate_ts, tmp_path: Path) -> None:
    """Saving then loading must reconstruct X and y exactly."""
    X, y = multivariate_ts
    file_path = tmp_path / "dataset.csv"

    save_multivariate_ts_as_csv(file_path, X, y)
    loaded_X, loaded_y = load_multivariate_ts_from_csv(file_path)

    assert np.array_equal(loaded_X, X)
    assert np.array_equal(loaded_y, y)


def test_csv_structure(tmp_path: Path) -> None:
    """The saved main CSV must follow the expected long-format schema."""
    X = np.arange(6).reshape(1, 3, 2)
    y = np.array([1])

    file_path = tmp_path / "ts.csv"
    save_multivariate_ts_as_csv(file_path, X, y)

    df = pd.read_csv(file_path)
    assert list(df.columns) == ["sample_id", "time_step", "feature_0", "feature_1"]
    assert df.shape == (3, 4)


def test_labels_saved_correctly(tmp_path: Path) -> None:
    """Label CSV must contain sample_id and label columns exactly once."""
    X = np.arange(12).reshape(2, 3, 2)
    y = np.array([7, 9])

    file_path = tmp_path / "abc.csv"
    save_multivariate_ts_as_csv(file_path, X, y)

    label_df = pd.read_csv(tmp_path / "abc_labels.csv")

    assert list(label_df.columns) == ["sample_id", "label"]
    assert np.array_equal(label_df["label"].values, y)


def test_load_infers_dimensions(tmp_path: Path, multivariate_ts) -> None:
    """load_multivariate_ts_from_csv must correctly infer samples, timesteps, features."""
    X, y = multivariate_ts
    file_path = tmp_path / "big.csv"

    save_multivariate_ts_as_csv(file_path, X, y)
    loaded_X, loaded_y = load_multivariate_ts_from_csv(file_path)

    assert loaded_X.shape == X.shape
    assert np.array_equal(loaded_y, y)


def test_load_with_missing_label_file(tmp_path: Path, multivariate_ts) -> None:
    """Missing label file should raise FileNotFoundError."""
    X, y = multivariate_ts
    file_path = tmp_path / "missing.csv"

    save_multivariate_ts_as_csv(file_path, X, y)
    (tmp_path / "missing_labels.csv").unlink()  # remove label file

    with pytest.raises(FileNotFoundError):
        load_multivariate_ts_from_csv(file_path)


def test_save_multivariate_ts_invalid_shapes(tmp_path: Path) -> None:
    """Arrays with invalid shape must raise ValueError from reshape."""
    X = np.arange(10)  # not 3D
    y = np.array([1])

    file_path = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        save_multivariate_ts_as_csv(file_path, X, y)
