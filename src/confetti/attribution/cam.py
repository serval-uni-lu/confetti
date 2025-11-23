import numpy as np
import keras
from keras.src.layers import Conv1D, Dense
from confetti.errors import CONFETTIConfigurationError


def cam(
    model,
    X_data: np.ndarray,
):
    """Compute Class Activation Maps (CAMs) [1]_. for 1D convolutional classifiers.

    This function computes class activation maps for a trained Keras model
    by combining the activations of the final ``Conv1D`` layer with the
    weights of the final ``Dense`` classifier layer. It highlights the
    temporal regions most influential for the model’s predicted class.

    Parameters
    ----------
    model : keras.Model
        A trained Keras model containing at least one ``Conv1D`` layer
        followed by a ``Dense`` classification layer.
    X_data : ndarray of shape (n_samples, timesteps, channels)
        Input multivariate time series for which CAMs will be computed.

    Returns
    -------
    ndarray of shape (n_samples, timesteps)
        The class activation map for each input instance.

    Raises
    ------
    CONFETTIConfigurationError
        If no ``Conv1D`` layer is found, if no ``Dense`` output layer is
        present, or if the number of filters in the final convolutional
        layer does not match the input dimensionality of the classifier layer.

    Note
    -----
    The CAM for a given instance is defined as::

        CAM(t) = Σ_k w[k, c] * A[k, t]

    where:
        - ``A[k, t]`` is the activation of filter ``k`` at timestep ``t``
          from the final convolutional layer.
        - ``w[:, c]`` are the weights connecting each filter to the
          predicted class ``c`` in the final dense layer.


    References
    ----------
    .. [1] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016).
        Learning deep features for discriminative localization.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2921-2929).
    """

    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    if not conv_layers:
        raise CONFETTIConfigurationError(
            message="No Conv1D layers found in model. CAM requires a Conv1D backbone.",
            param="model",
        )

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

        conv_out = conv_out.numpy()  # (1, timesteps, filters)
        predicted_class = np.argmax(predictions.numpy())
        filters = conv_out.shape[-1]

        # Sanity check: filters must match classifier input dimension
        if filters != w_k_c.shape[0]:
            raise CONFETTIConfigurationError(
                f"Mismatch between last Conv1D output channels ({filters}) "
                f"and classifier input features ({w_k_c.shape[0]})."
                "\nCAM requires these to match."
            )

        # Compute CAM(t) = sum_k (w_kc * A_k(t))
        cam_signal = np.dot(conv_out[0], w_k_c[:, predicted_class])
        cam_results.append(cam_signal)

    return np.array(cam_results)
