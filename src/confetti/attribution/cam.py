import numpy as np
from confetti.errors import CONFETTIConfigurationError
from confetti.utils._compat import is_torch_module, require_keras, require_torch


def cam(
    model,
    X_data: np.ndarray,
):
    """Compute Class Activation Maps (CAMs) [1]_. for 1D convolutional classifiers.

    This function computes class activation maps for a trained model
    by combining the activations of the final convolutional layer with the
    weights of the final classifier layer. It highlights the
    temporal regions most influential for the model's predicted class.

    Both Keras and PyTorch models are supported. For Keras models, the
    function inspects ``Conv1D`` and ``Dense`` layers. For PyTorch models,
    it inspects ``Conv1d`` and ``Linear`` layers via forward hooks.

    Parameters
    ----------
    model : keras.Model, torch.nn.Module, or TorchModelAdapter
        A trained model containing at least one 1-D convolutional layer
        followed by a linear classification layer.
    X_data : ndarray of shape (n_samples, timesteps, channels)
        Input multivariate time series for which CAMs will be computed.

    Returns
    -------
    ndarray of shape (n_samples, timesteps)
        The class activation map for each input instance.

    Raises
    ------
    CONFETTIConfigurationError
        If no convolutional layer is found, if no classifier layer is
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
    from confetti.adapters import TorchModelAdapter

    if isinstance(model, TorchModelAdapter) or is_torch_module(model):
        return _cam_torch(model, X_data)
    return _cam_keras(model, X_data)


def _cam_keras(model, X_data: np.ndarray) -> np.ndarray:
    """CAM implementation for Keras models."""
    keras = require_keras("CAM computation with a Keras model")
    from keras.src.layers import Conv1D, Dense

    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    if not conv_layers:
        raise CONFETTIConfigurationError(
            message="No Conv1D layers found in model. CAM requires a Conv1D backbone.",
            param="model",
        )

    last_conv_layer = conv_layers[-1]

    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if not dense_layers:
        raise CONFETTIConfigurationError(
            message="Model does not contain a final Dense classifier layer.",
            param="model",
        )

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


def _cam_torch(model_or_adapter, X_data: np.ndarray) -> np.ndarray:
    """CAM implementation for PyTorch models using forward hooks."""
    torch = require_torch("CAM computation with a PyTorch model")
    import torch.nn as nn

    from confetti.adapters import TorchModelAdapter

    # Unwrap adapter if needed
    if isinstance(model_or_adapter, TorchModelAdapter):
        torch_model = model_or_adapter.torch_model
    else:
        torch_model = model_or_adapter

    torch_model.eval()

    # Find last Conv1d and last Linear layers
    conv_layers = [m for m in torch_model.modules() if isinstance(m, nn.Conv1d)]
    if not conv_layers:
        raise CONFETTIConfigurationError(
            message="No Conv1d layers found in model. CAM requires a Conv1d backbone.",
            param="model",
        )

    linear_layers = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise CONFETTIConfigurationError(
            message="Model does not contain a final Linear classifier layer.",
            param="model",
        )

    last_conv = conv_layers[-1]
    classifier = linear_layers[-1]

    # PyTorch Linear.weight shape: (out_features, in_features)
    # Transpose to (n_filters, n_classes) to match Keras convention
    w_k_c = classifier.weight.detach().cpu().numpy().T

    # Register hook to capture conv output
    conv_output: dict = {}

    def hook_fn(module, input, output):
        conv_output["value"] = output

    handle = last_conv.register_forward_hook(hook_fn)

    cam_results = []

    try:
        with torch.no_grad():
            for series in X_data:
                if series.ndim == 2:
                    series = series[np.newaxis, ...]
                else:
                    series = series.reshape(1, -1, series.shape[-1])

                # PyTorch Conv1d expects (batch, channels, timesteps)
                tensor = torch.tensor(series, dtype=torch.float32).permute(0, 2, 1)
                predictions = torch_model(tensor)

                predicted_class = predictions.argmax(dim=1).item()

                # conv_output['value'] shape: (1, filters, timesteps)
                # Transpose to (1, timesteps, filters) to match Keras convention
                conv_out = conv_output["value"].cpu().numpy()
                conv_out = conv_out.transpose(0, 2, 1)

                filters = conv_out.shape[-1]
                if filters != w_k_c.shape[0]:
                    raise CONFETTIConfigurationError(
                        f"Mismatch between Conv1d output channels ({filters}) "
                        f"and classifier input features ({w_k_c.shape[0]})."
                        "\nCAM requires these to match."
                    )

                # Compute CAM(t) = sum_k (w_kc * A_k(t))
                cam_signal = np.dot(conv_out[0], w_k_c[:, predicted_class])
                cam_results.append(cam_signal)
    finally:
        handle.remove()

    return np.array(cam_results)
