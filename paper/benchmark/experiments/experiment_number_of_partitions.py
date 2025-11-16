from numpy import floating

from confetti import CONFETTI
from confetti.explainer.counterfactuals import CounterfactualResults, CounterfactualSet, Counterfactual
from confetti.utils import (
    load_multivariate_ts_from_csv,
    load_data,
)
from typing import List
from paper.CAM import compute_weights_cam
import pandas as pd
import numpy as np
from paper import config as cfg
import keras
import time
from pathlib import Path

keras.utils.disable_interactive_logging()


def get_sparsity(original: np.ndarray, counterfactuals: List[Counterfactual]):
    """Compute the mean sparsity across multiple counterfactuals."""
    sparsities: list[floating] = [
        np.mean(original.flatten() == cf.counterfactual.flatten()) for cf in counterfactuals
    ]
    return float(np.mean(sparsities))



def get_confidence(model,
                   counterfactual : List[Counterfactual],
                   timesteps : int,
                   channels : int,
                   original_label):
    confidences : list[floating] = [
        1
        - model.predict(cf.counterfactual.reshape(1, timesteps, channels))[0][
            original_label
        ]
        for cf in counterfactual
    ]
    return float(np.mean(confidences))


partitions = [2, 4, 8, 16, 32, 64]


def run_experiment(model_name: str = "fcn"):
    dataset = "ArticularyWordRecognition"
    # Load model
    model_path = cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
    model = keras.models.load_model(model_path)

    # Load the sample data
    sample_path = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
    X_samples, y_samples = load_multivariate_ts_from_csv(sample_path)
    X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
    training_weights : np.ndarray = compute_weights_cam(
        model=model,
        X_data=X_train,
        dataset=dataset,
        save_weights=False,
        data_type="training",
    )
    exp = CONFETTI(model_path=model_path)
    # Create a dict to store results
    results = {"Partitions": [], "Execution Time": [], "Sparsity": [], "Confidence": []}
    for partition in partitions:
        start_time = time.time()
        counterfactuals : CounterfactualResults = exp.generate_counterfactuals(
            instances_to_explain=X_samples,
            reference_data=X_train,
            reference_weights=training_weights,
            n_partitions=partition,
            processes=6,
            verbose=True,
        )
        end_time = time.time()
        execution_time = end_time - start_time
        # Calculate sparsity and confidence
        sparsities = []
        confidences = []
        for i in range(len(counterfactuals)):
            counterfactual_set : CounterfactualSet = counterfactuals[i]
            original_instance : np.ndarray = counterfactual_set.original_instance
            og_label = counterfactual_set.original_label
            sparsity = get_sparsity(
                original=original_instance, counterfactuals=counterfactual_set.all_counterfactuals
            )
            confidence = get_confidence(
                model=model,
                counterfactual=counterfactual_set.all_counterfactuals,
                timesteps=X_samples.shape[1],
                channels=X_samples.shape[2],
                original_label=og_label,
            )
            sparsities.append(sparsity)
            confidences.append(confidence)
        # Compute average sparsity and confidence
        avg_sparsity = np.mean(sparsities)
        avg_confidence = np.mean(confidences)
        # Append results to DataFrame
        results["Partitions"].append(partition)
        results["Execution Time"].append(execution_time)
        results["Sparsity"].append(avg_sparsity)
        results["Confidence"].append(avg_confidence)
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Save results to CSV
    results_file = Path.cwd() / f"{model_name}_partitions_results.csv"
    results_df.to_csv(results_file, index=False)


def main():
    # Run the experiment for the specified model
    run_experiment(model_name="resnet")


if __name__ == "__main__":
    main()
