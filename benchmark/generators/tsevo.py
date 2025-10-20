import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import time
from tqdm import tqdm
from pathlib import Path
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from confetti.explainer.utils import (
    load_data,
    load_multivariate_ts_from_csv,
    array_to_string,
)
import config as cfg

tf.keras.utils.disable_interactive_logging()


def run_tsevo_counterfactuals(model_name=None):
    datasets = cfg.DATASETS
    times = {}

    for dataset in tqdm(datasets, desc="Processing datasets"):
        model_path = cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
        model = keras.models.load_model(str(model_path))

        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=True)
        reference_labels = np.argmax(model.predict(X_train), axis=1)

        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        print(f"Loaded {dataset} samples from CSV: {X_samples.shape}")

        exp_model = TSEvo(
            model=model,
            data=(X_train, reference_labels),
            backend="TF",
            mode="time",
            epochs=100,
            verbose=0,
        )
        print("Explainer created.")

        results_df = pd.DataFrame(
            columns=["Solution", "Test Instance", "Original Label", "CE Label"]
        )

        start_time = time.time()
        for i, instance in enumerate(
            tqdm(X_samples, desc=f"Generating CEs for {dataset}")
        ):
            item = instance.reshape(1, *instance.shape)
            label = np.array([y_samples[i]])  # <-- Pass as class index

            cf_solution, label_cf = exp_model.explain(item, label)

            row_dict = {
                "Solution": [cf_solution],
                "Test Instance": i,
                "Original Label": label[0],
                "CE Label": np.argmax(label_cf),
            }
            row_df = pd.DataFrame(row_dict)
            results_df = pd.concat([results_df, row_df], ignore_index=True)

        end_time = time.time()
        times[dataset] = end_time - start_time

        results_df["Solution"] = results_df["Solution"].apply(array_to_string)
        results_directory = cfg.RESULTS_DIR / dataset
        results_directory.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(
            results_directory / f"tsevo_{dataset}_{model_name}_counterfactuals.csv",
            index=False,
        )
        print(f"Results for {dataset} saved to CSV.")

    results_directory = cfg.RESULTS_DIR
    times_df = pd.DataFrame(
        times.items(), columns=["Dataset", "Execution Time (seconds)"]
    )
    times_df.to_csv(
        results_directory / f"execution_time_{model_name}_tsevo.csv", index=False
    )
    print("Execution times saved.")


def main():
    run_tsevo_counterfactuals(model_name="resnet")


if __name__ == "__main__":
    main()
