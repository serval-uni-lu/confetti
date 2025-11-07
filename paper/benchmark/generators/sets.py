import numpy as np
import keras
import pandas as pd
import time
from tqdm import tqdm
from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF
from src.confetti.utils import (
    load_data,
    load_multivariate_ts_from_csv,
    array_to_string,
)
from paper import config as cfg

keras.utils.disable_interactive_logging()


def run_sets_counterfactuals(model_name=None):
    datasets = ["NATOPS"]
    times_file = cfg.RESULTS_DIR / f"execution_time_sets_{model_name}.csv"
    results_directory = cfg.RESULTS_DIR
    results_directory.mkdir(parents=True, exist_ok=True)

    # Initialize the CSV file with header if it doesn't exist
    if not times_file.exists():
        pd.DataFrame(columns=["Dataset", "Execution Time (seconds)"]).to_csv(
            times_file, index=False
        )

    for dataset in tqdm(datasets, desc="Processing datasets"):
        # load the model
        model_path = cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
        model = keras.models.load_model(model_path)

        # Sets needs one-hot encoding to be False because it does the One-Hot encoding internally
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
        reference_labels = np.argmax(model.predict(X_train), axis=1)

        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        print(f"Loaded {dataset} samples from CSV: {X_samples.shape}")

        # Determine time series length (assuming shape is [n_samples, n_channels, series_length])
        series_length = X_train.shape[2]
        min_shapelet_len = max(3, series_length // 10)
        max_shapelet_len = max(min_shapelet_len + 1, series_length // 2)

        print(
            f"series_length: {series_length}, min_shapelet_len: {min_shapelet_len}, max_shapelet_len: {max_shapelet_len}"
        )
        exp_model = SETSCF(
            model,
            (X_train, reference_labels),
            backend="TF",
            mode="time",
            min_shapelet_len=min_shapelet_len,
            max_shapelet_len=max_shapelet_len,
            time_contract_in_mins_per_dim=0.5,
            fit_shapelets=False,
        )

        exp_model.fit(occlusion_threshhold=0.6, remove_multiclass_shapelets=True)
        print("Explainer created and fitted.")

        results_sets = pd.DataFrame(
            columns=["Solution", "Test Instance", "Original Label", "CE Label"]
        )

        start_time = time.time()
        for i, instance in enumerate(
            tqdm(X_samples, desc=f"Generating CEs for {dataset}")
        ):
            item = X_samples[i]

            try:
                cf_explanation, label_cf = exp_model.explain(item, target=None)
                if cf_explanation is None:
                    print(f"CF explanation is None for instance {i}. Skipping.")
                    continue
            except AssertionError as e:
                if "Pertubed instance is identical to the original instance" in str(e):
                    print(f"No CF found for instance {i}. Appending None.")
                    continue
                else:
                    raise  # Re-raise if it's a different assertion error

            row_dict = {
                "Solution": [cf_explanation],
                "Test Instance": i,
                "Original Label": y_samples[i],
                "CE Label": label_cf,
            }
            row_df = pd.DataFrame(row_dict)
            results_sets = pd.concat([results_sets, row_df], ignore_index=True)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save results for the current dataset
        results_sets["Solution"] = results_sets["Solution"].apply(array_to_string)
        dataset_result_dir = cfg.RESULTS_DIR / dataset
        dataset_result_dir.mkdir(parents=True, exist_ok=True)
        results_sets.to_csv(
            dataset_result_dir / f"sets_{dataset}_{model_name}_counterfactuals.csv",
            index=False,
        )
        print(f"Results for {dataset} saved to CSV.")

        # Append execution time to the shared CSV
        new_time_row = pd.DataFrame(
            [[dataset, elapsed_time]], columns=["Dataset", "Execution Time (seconds)"]
        )
        new_time_row.to_csv(times_file, mode="a", header=False, index=False)
        print(f"Execution time for {dataset} saved.")


def main():
    run_sets_counterfactuals(model_name="resnet")


if __name__ == "__main__":
    main()
