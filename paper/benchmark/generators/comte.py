import numpy as np
import keras
import pandas as pd
import time
from tqdm import tqdm
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from src.confetti.utils import (
    load_data,
    load_multivariate_ts_from_csv,
    array_to_string,
)
from paper import config as cfg

keras.utils.disable_interactive_logging()


def run_comte_counterfactuals(model_name=None):
    datasets = cfg.DATASETS
    results_directory = cfg.RESULTS_DIR
    results_directory.mkdir(parents=True, exist_ok=True)

    time_file = results_directory / f"execution_time_comte_{model_name}.csv"

    # Initialize time file if it doesn't exist
    if not time_file.exists():
        pd.DataFrame(columns=["Dataset", "Execution Time (seconds)"]).to_csv(
            time_file, index=False
        )

    for dataset in tqdm(datasets, desc="Processing datasets"):
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=True)

        model_path = cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
        model = keras.models.load_model(str(model_path))

        # Reference labels are the predicted labels for the training set. NOT the ground truth labels.
        reference_labels = np.argmax(model.predict(X_train), axis=1)

        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        print(f"Loaded {dataset} samples from CSV: {X_samples.shape}")

        exp_model = COMTECF(
            model=model,
            data=(X_train, reference_labels),
            mode="time",
            backend="TF",
            method="opt",
            number_distractors=1,
            max_attempts=100,
            max_iter=100,
        )

        results_comte = pd.DataFrame(
            columns=["Solution", "Test Instance", "Original Label", "CE Label"]
        )
        print("Explainer created.")

        start_time = time.time()
        for i, instance in enumerate(
            tqdm(X_samples, desc=f"Generating CEs for {dataset}")
        ):
            item = X_samples[i].reshape(1, *X_samples[i].shape)
            y_target = model.predict(item)

            try:
                exp = exp_model.explain(item)
                array, label = exp

                org_label = np.argmax(y_target)
                cf_label = label[0]
                exp = np.array(array)

                results_dict = {
                    "Solution": [exp[0]],
                    "Test Instance": i,
                    "Original Label": org_label,
                    "CE Label": cf_label,
                }
                results_df = pd.DataFrame(results_dict)
                results_comte = pd.concat(
                    [results_comte, results_df], ignore_index=True
                )

            except AttributeError as e:
                if "'list' object has no attribute 'query'" in str(e):
                    print(
                        f"Skipping instance {i}: KDTree could not be built (likely due to class imbalance)."
                    )
                else:
                    print(f"Unexpected error for instance {i}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save counterfactuals
        results_comte["Solution"] = results_comte["Solution"].apply(array_to_string)
        dataset_result_dir = cfg.RESULTS_DIR / dataset
        dataset_result_dir.mkdir(parents=True, exist_ok=True)
        results_comte.to_csv(
            dataset_result_dir / f"comte_{dataset}_{model_name}_counterfactuals.csv",
            index=False,
        )
        print(f"Results for {dataset} saved to CSV.")

        # Append execution time to file
        time_row = pd.DataFrame(
            [[dataset, elapsed_time]], columns=["Dataset", "Execution Time (seconds)"]
        )
        time_row.to_csv(time_file, mode="a", header=False, index=False)
        print(f"Execution time for {dataset} saved.")


def main():
    run_comte_counterfactuals(model_name="resnet")


if __name__ == "__main__":
    main()
