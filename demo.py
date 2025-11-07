from paper import config as cfg
import keras
import paper.CAM.class_activation_map as cam
from confetti import CONFETTI, CounterfactualResults
from confetti.utils import  load_data, load_multivariate_ts_from_csv, array_to_string
import numpy as np
import pandas as pd
from paper.benchmark.evaluations.evaluator import Evaluator

keras.utils.disable_interactive_logging()


def create_standard_results_dataframe(
    counterfactuals: CounterfactualResults,
    test_instances: np.ndarray,
    nun_instances: np.ndarray
) -> pd.DataFrame:
    """Create the standard DataFrame used by the Evaluator from CounterfactualResults."""
    df = counterfactuals.to_dataframe()
    df["counterfactual"] = df["counterfactual"].apply(array_to_string)

    df["Test Instance"] = [
        next((i for i, instance in enumerate(test_instances) if np.array_equal(original, instance)), None)
        for original in df["original_instance"]
    ]

    df["NUN Instance"] = [
        next((i for i, instance in enumerate(nun_instances) if np.array_equal(nun, instance)), None)
        for nun in df["nearest_unlike_neighbour"]
    ]

    df = df.rename(columns={"label": "CE Label", "counterfactual": "Solution"})
    return df[["Solution", "Test Instance", "NUN Instance", "CE Label"]].reset_index(drop=True)


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
    X_samples = X_samples[0:5]
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

    counterfactuals: CounterfactualResults = explainer.generate_counterfactuals(
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
        verbose=True,
        processes=6,
        save_counterfactuals=False,
    )

    results_df = create_standard_results_dataframe(
        counterfactuals=counterfactuals,
        test_instances=X_samples,
        nun_instances=X_train,
    )
    results_df.to_csv("temp.csv", index=False)

    ev = Evaluator()
    metrics, summary = ev.evaluate_from_csv(explainer="Example",
                                            dataset="ArticularyWordRecognition",
                                            model_name="resnet",
                                            optional_path="temp.csv")
    print(f"Summary:\n{summary}")







def main():
    run_demo()


if __name__ == "__main__":
    main()
