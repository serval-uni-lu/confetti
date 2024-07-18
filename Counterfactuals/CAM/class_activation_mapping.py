import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def visualize_cam(weights, instance_index):
    #Normalize weights
    scaler = MinMaxScaler()
    normalized_weights = scaler.fit_transform(weights)

    #Get Sample
    sample = normalized_weights[instance_index]

    #Plotting
    plt.figure(figsize=(10, 6)) # Set size
    sns.set_theme(style="whitegrid")  # Setting style
    sns.lineplot(data=sample, color='blue', alpha=.7, linewidth=2)
    plt.title(f'Class Activation Map for Instance {instance_index}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Weight')
    plt.ylim(0, 1)  # Setting y-axis limits
    plt.show()

def compute_weights_cam(model, X_data, dataset, save_weights, weights_directory, data_type='testing'):
    """
    Compute class activation maps (CAMs) for the given data.

    Parameters:
        model (keras.Model): The trained model.
        X_data (numpy.ndarray): The input data (either training or testing).
        dataset (str): The name of the dataset.
        save_weights (bool): Whether to save the computed weights to a file.
        weights_directory (str): Where to save the computed weights.
        data_type (str): The type of data ('training' or 'testing').

    Returns:
        numpy.ndarray: The computed class activation maps.
    """
    w_k_c = model.layers[-1].get_weights()[0]

    new_input_layer = model.inputs
    # Output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    weights = []
    for i, ts in enumerate(X_data):
        if ts.shape[-1] == 1:  # Check dimensions
            ts = ts.reshape(1, -1, 1)  # Reshape for Univariate
        else:
            ts = ts.reshape(-1, ts.shape[0], ts.shape[1])  # Reshape for Multivariate
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted)

        cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
        for k, w in enumerate(w_k_c[:, pred_label]):
            cas += w * conv_out[0, :, k]
        weights.append(cas)
    weights = np.array(weights)

    if save_weights:
        np.save(f"{weights_directory}{dataset}_{data_type}_weights.npy", weights)

    return weights
