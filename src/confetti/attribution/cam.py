import numpy as np
import keras
from keras.src.layers import Conv1D, Dense

def cam(
    model,
    X_data : np.ndarray,
):
    """
    Compute Class Activation Maps (CAMs) for a given Keras model and dataset.

    This version automatically:
      - Finds the last Conv1D layer (feature map source)
      - Finds the final Dense classifier layer
      - Ensures channel counts match
      - Computes CAM without any architecture-specific assumptions

    Parameters:
        model (keras.Model): A trained Keras model.
        X_data (numpy.ndarray): Input time series (n_samples, timesteps, channels).
        dataset (str): Optional, unused but kept for compatibility.
        data_type (str): Optional, kept for compatibility.

    Returns:
        numpy.ndarray: CAM weights of shape (n_samples, timesteps).
    """


    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    if not conv_layers:
        raise ValueError("No Conv1D layers found in model. CAM requires a Conv1D backbone.")

    last_conv_layer = conv_layers[-1]


    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if not dense_layers:
        raise ValueError("Model does not contain a final Dense classifier layer.")

    classifier_layer = dense_layers[-1]

    # Extract classifier weights: shape (n_filters, n_classes)
    w_k_c = classifier_layer.get_weights()[0]


    cam_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, classifier_layer.output],
    )


    cam_results = []

    for series in X_data:
        if series.ndim == 2:
            series = series[np.newaxis, ...]
        else:
            series = series.reshape(1, -1, series.shape[-1])

        conv_out, predictions = cam_model(series)

        conv_out = conv_out.numpy()      # (1, timesteps, filters)
        predicted_class = np.argmax(predictions.numpy())
        filters = conv_out.shape[-1]

        # Sanity check: filters must match classifier input dimension
        if filters != w_k_c.shape[0]:
            raise ValueError(
                f"Mismatch between last Conv1D output channels ({filters}) "
                f"and classifier input features ({w_k_c.shape[0]})."
                "\nCAM requires these to match."
            )

        # Compute CAM(t) = sum_k (w_kc * A_k(t))
        cam_signal = np.dot(conv_out[0], w_k_c[:, predicted_class])
        cam_results.append(cam_signal)

    return np.array(cam_results)