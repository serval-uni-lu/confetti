import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import time
from tqdm import tqdm
from pathlib import Path
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from confetti.explainer.utils import load_data, load_multivariate_ts_from_csv
import config as cfg
from keras.utils import to_categorical

tf.keras.utils.disable_interactive_logging()


def run_tsevo_counterfactuals():
    datasets = cfg.DATASETS
    times = {}

    for dataset in tqdm(datasets, desc='Processing datasets'):
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=True)

        model_path = cfg.TRAINED_MODELS_DIR / dataset / f'{dataset}_fcn.keras'
        model = keras.models.load_model(str(model_path))

        sample_file = f"{cfg.DATA_DIR}/{dataset}_samples.csv"
        X_samples, y_samples_raw = load_multivariate_ts_from_csv(sample_file)
        print(f"Loaded {dataset} samples from CSV: {X_samples.shape}")

        # Infer num_classes from y_train (one-hot encoded) or y_train.max() + 1
        if len(y_train.shape) > 1:
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))

        # One-hot for potential later usage
        y_samples_onehot = to_categorical(y_samples_raw, num_classes=num_classes)

        exp_model = TSEvo(model=model,
                          data=(X_test, y_test),
                          backend='TF',
                          mode='time',
                          epochs=100,
                          verbose=0)
        print("Explainer created.")

        results_df = pd.DataFrame(columns=["Solution", "Test Instance", "Original Label", "CE Label"])

        start_time = time.time()
        for i, instance in enumerate(tqdm(X_samples, desc=f"Generating CEs for {dataset}")):
            item = instance.reshape(1, *instance.shape)
            label = np.array([y_samples_raw[i]])  # <-- Pass as class index

            cf_solution, label_cf = exp_model.explain(item, label)

            row_dict = {
                'Solution': [cf_solution],
                'Test Instance': i,
                'Original Label': label[0],
                'CE Label': np.argmax(label_cf)
            }
            row_df = pd.DataFrame(row_dict)
            results_df = pd.concat([results_df, row_df], ignore_index=True)

        end_time = time.time()
        times[dataset] = end_time - start_time

        results_directory = cfg.RESULTS_DIR / dataset
        results_directory.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_directory / f'tsevo_{dataset}_fcn_counterfactuals.csv', index=False)
        print(f"Results for {dataset} saved to CSV.")

    results_directory = cfg.RESULTS_DIR
    times_df = pd.DataFrame(times.items(), columns=['Dataset', 'Execution Time (seconds)'])
    times_df.to_csv(results_directory / 'execution_time_fcn_tsevo.csv', index=False)
    print("Execution times saved.")

def main():
    run_tsevo_counterfactuals()

if __name__ == "__main__":
    main()
