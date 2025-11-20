import os
from paper import config as cfg
from confetti.utils import (
    load_data,
    load_multivariate_ts_from_csv,
)
from paper.utils import get_samples


if __name__ == "__main__":
    # Current directory
    output_dir = os.path.join(os.path.dirname(__file__))

    for dataset in cfg.DATASETS:
        file_path = os.path.join(output_dir, f"{dataset}_resnet_samples.csv")

        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found! Run `generate_samples.py` first.")
            continue

        # Load original data to get expected shape
        if dataset not in cfg.NUMBER_OF_SAMPLES_PER_CLASS:
            X_train, X_expected, y_train, y_expected = load_data(dataset, one_hot=False)
        else:
            X_expected, y_expected = get_samples(dataset, one_hot=False)

        # Load dataset from CSV
        X_loaded, y_loaded = load_multivariate_ts_from_csv(file_path)

        # Validate shape
        if X_loaded.shape == X_expected.shape and y_loaded.shape == y_expected.shape:
            print(f"✅ {dataset}: Data shape is correct! {X_loaded.shape}")
        else:
            print(
                f"❌ {dataset}: Shape mismatch! Expected {X_expected.shape}, but got {X_loaded.shape}"
            )
