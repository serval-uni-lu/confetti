import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs

def ucr_data_loader(dataset):
    
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    y_train, y_test = label_encoder(y_train,y_test)
    
    return X_train, y_train, X_test, y_test


def label_encoder(training_labels, testing_labels):
    
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(training_labels)
    y_test = le.transform(testing_labels)
    
    return y_train, y_test

def reshape_ts(X_train, X_test):
    #This will make X_train from tslearn.dataset from (rows, timesteps, columns) to (rows, columns, timesteps)
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0],X_train.shape[2],X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test,(X_test.shape[0], X_test.shape[2],X_test.shape[1]))

    return X_train_reshaped,X_test_reshaped

def visualize_series(series):
    # Transpose the series to swap dimensions
    series_reshaped = series.T 

    # Determine the number of dimensions (subplots) based on the input series
    num_dimensions = series_reshaped.shape[0]

    # Create a plot with dynamic number of subplots for the time series
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions))  # Adjust the height based on number of dimensions

    # Check if there is more than one dimension to avoid indexing errors
    if num_dimensions > 1:
        for i, ax in enumerate(axes):
            ax.plot(series_reshaped[i])
            ax.set_title(f"Dimension {i+1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
    else:  # Handle the case with only one dimension
        axes.plot(series_reshaped[0])
        axes.set_title("Dimension 1")
        axes.set_xlabel("Time")
        axes.set_ylabel("Value")

    plt.tight_layout()
    plt.show()

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
