import numpy as np
import tensorflow as tf
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import config as cfg


def compute_weights_cam(
    model,
    X_data,
    dataset,
    save_weights=True,
    weights_directory=None,
    data_type="testing",
):
    """
    Compute class activation maps (CAMs) for the given data.

    Parameters:
        model: The trained model (Keras, PyTorch, or TensorFlow).
        X_data (numpy.ndarray or torch.Tensor): The input data.
        dataset (str): The name of the dataset.
        save_weights (bool): Whether to save the computed weights to a file.
        weights_directory (str or Path, optional): Where to save the computed weights. Defaults to 'models/trained_models/cam_weights/'.
        data_type (str): The type of data ('training' or 'testing').

    Returns:
        numpy.ndarray: The computed class activation maps.
    """

    # Default directory: models/trained_models/{dataset}
    if weights_directory is None:
        weights_directory = cfg.CAM_WEIGHTS_DIR / dataset
    else:
        weights_directory = Path(weights_directory)

    weights_directory.mkdir(parents=True, exist_ok=True)

    weights = []

    # Detect model type
    if isinstance(
        model, tf.keras.Model
    ):  # TensorFlow/Keras Model (Keras 3 & TensorFlow 2.18.0)
        last_conv_layer = model.get_layer(
            index=-3
        )  # Assuming last conv layer is 3rd from the end
        classifier_layer = model.get_layer(index=-1)  # Output layer
        w_k_c = classifier_layer.weights[0].numpy()

        model_cam = tf.keras.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, classifier_layer.output],
        )

        for ts in X_data:
            ts = (
                ts[np.newaxis, ...] if ts.ndim == 2 else ts.reshape(1, -1, ts.shape[-1])
            )
            conv_out, predicted = model_cam(ts)
            pred_label = np.argmax(predicted)

            cas = np.zeros(shape=(conv_out.shape[1]), dtype=np.float32)
            for k, w in enumerate(w_k_c[:, pred_label]):
                cas += w * conv_out[0, :, k]
            weights.append(cas)

    elif isinstance(model, torch.nn.Module):  # PyTorch 2.6.0 Model
        model.eval()
        last_conv_layer = list(model.children())[
            -3
        ]  # Assuming the last conv layer is the 3rd from the end
        classifier_layer = list(model.children())[-1]

        def hook_fn(module, input, output):
            return output

        activation = None
        hook = last_conv_layer.register_forward_hook(
            lambda module, input, output: setattr(module, "activation", output)
        )

        with torch.no_grad():
            for ts in X_data:
                ts = (
                    torch.tensor(ts, dtype=torch.float32).unsqueeze(0)
                    if len(ts.shape) == 2
                    else torch.tensor(ts, dtype=torch.float32).unsqueeze(0)
                )
                logits = model(ts)
                pred_label = torch.argmax(logits, dim=1).item()

                conv_out = last_conv_layer.activation  # Extract activation
                w_k_c = classifier_layer.weight[pred_label].detach().numpy()

                cas = np.zeros(conv_out.shape[2], dtype=np.float32)
                for k, w in enumerate(w_k_c):
                    cas += w * conv_out[0, k, :].numpy()
                weights.append(cas)

        hook.remove()  # Remove hook to prevent memory leaks
    else:
        raise TypeError(
            "Unsupported model type. Provide a Keras, TensorFlow, or PyTorch model."
        )

    weights = np.array(weights)

    if save_weights:
        file_path = weights_directory / f"{dataset}_{data_type}_weights.npy"
        np.save(file_path, weights)

    return weights


def visualize_cam(weights: np.ndarray, instance_index: int):
    """
    Visualizes the Class Activation Map (CAM) for a specific instance.

    Parameters:
        weights (np.ndarray): A 2D array containing CAM weights for multiple instances.
        instance_index (int): The index of the instance to visualize.

    Returns:
        None. Displays a line plot of the CAM for the specified instance.
    """

    # Normalize weights
    scaler = MinMaxScaler()
    normalized_weights = scaler.fit_transform(weights)

    # Get Sample
    sample = normalized_weights[instance_index]

    # Plotting
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.set_theme(style="whitegrid")  # Set Seaborn theme
    sns.lineplot(data=sample, color="#018575", alpha=0.7, linewidth=2)

    # Titles and labels
    plt.title(f"Class Activation Map for Instance {instance_index}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Weight")
    plt.ylim(0, 1)  # Setting y-axis limits

    # Show plot
    plt.show()
