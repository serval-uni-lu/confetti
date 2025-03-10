import numpy as np
from sklearn import preprocessing
import keras
from sktime.datasets import load_UCR_UEA_dataset

def convert_string_to_array(data_string, timesteps, channels):
    # Remove the square brackets and newline characters
    cleaned_data = data_string.replace('[', '').replace(']', '').replace('\n', '')

    # Convert the cleaned string to a numpy array
    array_data = np.fromstring(cleaned_data, sep=' ')

    # Reshape the array into the correct dimensions (100, 6)
    # Ensure the correct total number of elements before reshaping
    if array_data.size == timesteps * channels:  # 100 * 6
        final_array = array_data.reshape(timesteps, channels)
        return final_array
    else:
        raise ValueError(f"Data does not match expected size ({timesteps}, {channels})")

def load_data(dataset:str, encode_labels=True):

    # Data will load with shape (instances, dimensions, timesteps)
    X_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset(dataset, split="test", return_type="numpy3d")

    # Reshape data to (instances, timesteps, dimensions)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    if encode_labels:
        # Encode
        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # One Hot the labels for the CNN
        y_train, y_test = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_test)

    print("Shape:", X_train.shape)

    return X_train, X_test, y_train, y_test

