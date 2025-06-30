import os
import keras
from confetti.explainer.utils import load_data, save_multivariate_ts_as_csv, get_predicted_samples
import config as cfg
import numpy as np


if __name__ == "__main__":
    #Current directory
    output_dir = os.path.join(os.path.dirname(__file__))
    datasets = cfg.DATASETS
    for dataset in datasets:
        model = keras.models.load_model(
            os.path.join(cfg.TRAINED_MODELS_DIR, dataset, f"{dataset}_resnet.keras")
        )
        if dataset not in cfg.NUMBER_OF_SAMPLES_PER_CLASS:
            X_train, X_samples, y_train, y_test = load_data(dataset, one_hot=False)
            y_samples = np.argmax(model.predict(X_samples), axis=1)
        else:
            X_samples, y_samples = get_predicted_samples(dataset, model, one_hot=False)

        file_path = os.path.join(output_dir, f"{dataset}_resnet_samples.csv")
        save_multivariate_ts_as_csv(file_path, X_samples, y_samples)
