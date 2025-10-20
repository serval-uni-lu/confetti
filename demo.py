import config as cfg
import keras
import warnings
import confetti.CAM.class_activation_map as cam
from confetti.explainer.confetti_explainer import CONFETTI
from confetti.explainer.utils import (
    load_data,
    load_multivariate_ts_from_csv,
    array_to_string,
)
import tensorflow as tf
import pandas as pd


def run_demo(
    model_name: str = "fcn",
    optimize_confidence: bool = True,
    optimize_sparsity: bool = True,
    optimize_proximity: bool = True,
    proximity_distance: str = "euclidean",
    dtw_window: int = None,
    dataset: str = "ArticularyWordRecognition",
):
    # Load samples and model
    sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
    model_path = str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras")

    X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
    model = keras.models.load_model(model_path)

    # Load reference data
    X_train, _, y_train, _ = load_data(dataset, one_hot=False)

    training_weights = cam.compute_weights_cam(
        model=model,
        X_data=X_train,
        dataset=dataset,
        save_weights=False,
        data_type="training",
    )

    explainer = CONFETTI(model_path=model_path)

    number_partitions = (
        int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)
    )

    _, ces_optimized = explainer.parallelized_counterfactual_generator(
        instances_to_explain=X_samples,
        reference_data=X_train,
        reference_weights=training_weights,
        alpha=0.5,
        theta=0.51,
        optimize_confidence=optimize_confidence,
        optimize_sparsity=optimize_sparsity,
        optimize_proximity=optimize_proximity,
        proximity_distance=proximity_distance,
        dtw_window=dtw_window,
        processes=6,
        verbose=False,
    )

    ces_optimized["Solution"] = ces_optimized["Solution"].apply(array_to_string)

    experiment_label = "".join(
        [
            "C" if optimize_confidence else "",
            "S" if optimize_sparsity else "",
            "P" if optimize_proximity else "",
        ]
    )

    # Save results to CSV in this directory
    # ces_optimized.to_csv(f"demo_{dataset}_{model_name}_{experiment_label}.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    distance = "dtw"  # user choice
    import importlib

    try:
        metrics_module = importlib.import_module("tslearn.metrics")
        # getattr fetches function by name if it exists
        metric_fn = getattr(metrics_module, distance)
        print("Success!")
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Metric '{distance}' not found in tslearn. "
            "Make sure tslearn is installed and the metric exists."
        ) from e


if __name__ == "__main__":
    main()
