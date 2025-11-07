"""
Script to obtain the average execution time of CONFETTI for each dataset to evaluate how the MTS dimensions
affect the execution time of the counterfactual generation process.
"""

from paper import config as cfg
import keras
import warnings
import paper.CAM.class_activation_map as cam
from confetti import CONFETTI
from confetti.utils import load_data, load_multivariate_ts_from_csv
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

keras.utils.disable_interactive_logging()
warnings.filterwarnings("ignore", category=FutureWarning)

# Dataset properties
ts_lengths = {
    "ArticularyWordRecognition": 144,
    "BasicMotions": 100,
    "Epilepsy": 207,
    "ERing": 65,
    "Libras": 45,
    "NATOPS": 51,
    "RacketSports": 30,
}
ts_dimensions = {
    "ArticularyWordRecognition": 9,
    "BasicMotions": 6,
    "Epilepsy": 3,
    "ERing": 4,
    "Libras": 2,
    "NATOPS": 24,
    "RacketSports": 6,
}


def run_execution_time_experiment(model_name="fcn", alpha=0.5, theta=0.51):
    results = []

    for dataset in cfg.DATASETS:
        print(f"Running {dataset}...")
        # Load model and data
        model_path = str(
            cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
        )
        model = keras.models.load_model(model_path)
        X_train, _, _, _ = load_data(dataset, one_hot=False)

        # Load precomputed samples for this dataset-model
        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, _ = load_multivariate_ts_from_csv(sample_file)

        # CAM weights
        training_weights : np.ndarray = cam.compute_weights_cam(
            model=model,
            X_data=X_train,
            dataset=dataset,
            save_weights=False,
            data_type="training",
        )

        instance_times = []
        for i in tqdm(
            range(len(X_samples)), desc=f"{dataset} - Timing instances", leave=False
        ):
            X_i = X_samples[i : i + 1]  # Keep shape (1, T, D)

            ce = CONFETTI(model_path=model_path)

            start = time.time()
            _ = ce.generate_counterfactuals(
                instances_to_explain=X_i,
                reference_data=X_train,
                reference_weights=training_weights,
                n_partitions=2,
                alpha=alpha,
                theta=theta,
                verbose=True,
            )
            elapsed = time.time() - start
            instance_times.append(elapsed)

        avg_time = np.mean(instance_times)
        results.append(
            {
                "Dataset": dataset,
                "Length": ts_lengths.get(dataset),
                "Dimension": ts_dimensions.get(dataset),
                "Average Execution Time": avg_time,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("execution_time_average.csv", index=False)


if __name__ == "__main__":
    run_execution_time_experiment()
