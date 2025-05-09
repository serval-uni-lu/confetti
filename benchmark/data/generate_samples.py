import os
import pandas as pd
from confetti.explainer.utils import load_data, get_samples, save_multivariate_ts_as_csv
import config as cfg


if __name__ == "__main__":
    #Current directory
    output_dir = os.path.join(os.path.dirname(__file__))
    datasets = ['Libras']
    for dataset in datasets:
        if dataset not in cfg.NUMBER_OF_SAMPLES_PER_CLASS:
            X_train, X_samples, y_train, y_samples = load_data(dataset, one_hot=False)
        else:
            X_samples, y_samples = get_samples(dataset, one_hot=False)

        file_path = os.path.join(output_dir, f"{dataset}_samples.csv")
        save_multivariate_ts_as_csv(file_path, X_samples, y_samples)
