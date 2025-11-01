from src.confetti import CONFETTI
from src.confetti.utils import (
    load_multivariate_ts_from_csv,
    load_data,
)
from paper.CAM import compute_weights_cam
import pandas as pd
import numpy as np
from paper import config as cfg
import keras
import time
from pathlib import Path


def get_sparsity(original: pd.DataFrame, counterfactual: pd.DataFrame):
    return np.mean(original.flatten() == counterfactual.flatten())


def get_confidence(model, counterfactual, timesteps, channels, original_label):
    return (
        1
        - model.predict(counterfactual.reshape(1, timesteps, channels))[0][
            original_label
        ]
    )


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
    training_weights = compute_weights_cam(
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
        ces_naive, ces_optimized = exp.parallelized_counterfactual_generator(
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
        for i in range(len(ces_optimized)):
            counterfactual = ces_optimized["Solution"].iloc[i]
            original_instance = X_samples[ces_optimized["Test Instance"].iloc[i]]
            og_label = y_samples[ces_optimized["Test Instance"].iloc[i]]
            sparsity = get_sparsity(
                original=original_instance, counterfactual=counterfactual
            )
            confidence = get_confidence(
                model=model,
                counterfactual=counterfactual,
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
