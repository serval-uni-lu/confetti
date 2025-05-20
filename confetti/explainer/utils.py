import numpy as np
from sklearn import preprocessing
import keras
from sktime.datasets import load_UCR_UEA_dataset
import config as cfg
import pandas as pd

def convert_string_to_array(data_string, timesteps, channels):
    # Remove the square brackets and newline characters
    cleaned_data = data_string.replace('[', '').replace(']', '').replace('\n', '')

    # Convert the cleaned string to a numpy array
    array_data = np.fromstring(cleaned_data, sep=' ')

    # Reshape the array into the correct dimensions
    # Ensure the correct total number of elements before reshaping
    if array_data.size == timesteps * channels:
        final_array = array_data.reshape(timesteps, channels)
        return final_array
    else:
        raise ValueError(f"Data does not match expected size ({timesteps}, {channels})")

def load_data(dataset:str, one_hot=True):

    # Data will load with shape (instances, dimensions, timesteps)
    X_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset(dataset, split="test", return_type="numpy3d")

    # Reshape data to (instances, timesteps, dimensions)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    if one_hot:
        # One Hot the labels for the CNN
        y_train, y_test = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_test)

    print("Shape:", X_train.shape)

    return X_train, X_test, y_train, y_test

def get_samples(dataset, one_hot=False):
    X_train, X_test, y_train, y_test = load_data(dataset, one_hot=one_hot)  # Pass one_hot flag

    # Get available classes (based on non-one-hot labels)
    available_classes = np.unique(np.argmax(y_train, axis=1) if one_hot else y_train)

    # Lists to store selected samples
    X_test_samples = []
    y_test_samples = []

    for cls in available_classes:
        # Find indices of the samples belonging to the class
        if one_hot:
            test_indices = np.where(np.argmax(y_test, axis=1) == cls)[0]
        else:
            test_indices = np.where(y_test == cls)[0]

        if test_indices.size > 0:
            number_samples = cfg.NUMBER_OF_SAMPLES_PER_CLASS[dataset]
            selected_test_indices = np.random.choice(test_indices, size=number_samples, replace=False)

            X_test_samples.extend(X_test[selected_test_indices])
            y_test_samples.extend(y_test[selected_test_indices])

    # Convert lists back to numpy arrays
    X_test_samples = np.array(X_test_samples)
    y_test_samples = np.array(y_test_samples)

    return X_test_samples, y_test_samples

def save_multivariate_ts_as_csv(file_path, X, y):
    """
    Save a multivariate time series dataset to CSV in a long format.

    :param file_path: Path to save the CSV file.
    :param X: 3D NumPy array of shape (n_samples, n_time_steps, n_features).
    :param y: 1D array of labels.
    """
    n_samples, n_time_steps, n_features = X.shape
    reshaped_X = X.reshape(n_samples * n_time_steps, n_features)  # Flatten time dimension
    sample_ids = [[i] * n_time_steps for i in range(n_samples)]  # Assign sample ID
    time_steps = [list(range(n_time_steps)) for _ in range(n_samples)]  # Assign time step index

    df = pd.DataFrame(reshaped_X, columns=[f"feature_{i}" for i in range(n_features)])
    df.insert(0, "sample_id", sum(sample_ids, []))
    df.insert(1, "time_step", sum(time_steps, []))

    # Save labels separately to ensure no misalignment
    labels_df = pd.DataFrame({"sample_id": range(n_samples), "label": y})

    df.to_csv(file_path, index=False)
    labels_df.to_csv(file_path.replace(".csv", "_labels.csv"), index=False)
    print(f"Saved dataset to {file_path} and labels to {file_path.replace('.csv', '_labels.csv')}")

def load_multivariate_ts_from_csv(file_path):
    """
    Load a multivariate time series dataset from CSV and reshape it back to its original form.

    :param file_path: Path to the CSV file containing the dataset.
    :return: (X, y) where X is a 3D NumPy array (n_samples, n_time_steps, n_features),
             and y is a 1D array of labels.
    """
    df = pd.read_csv(file_path)
    labels_df = pd.read_csv(file_path.replace(".csv", "_labels.csv"))

    # Extract number of samples, time steps, and features
    n_samples = labels_df.shape[0]
    n_time_steps = df["time_step"].nunique()
    n_features = df.shape[1] - 2  # Subtract sample_id and time_step columns

    # Reshape back to (n_samples, n_time_steps, n_features)
    X = df.drop(columns=["sample_id", "time_step"]).values
    X = X.reshape(n_samples, n_time_steps, n_features)

    y = labels_df["label"].values  # Load labels

    return X, y

def array_to_string(arr):
    return ' '.join(map(str, arr.flatten()))



