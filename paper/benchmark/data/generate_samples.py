import os
import keras
import joblib
from confetti.utils import (
    load_data,
    save_multivariate_ts_as_csv,
    get_predicted_samples,
)
from paper import config as cfg
import numpy as np
from tqdm import tqdm

def create_samples(model_name: str):
    # Current directory
    output_dir = os.path.join(os.path.dirname(__file__))
    datasets = cfg.DATASETS
    for dataset in tqdm(datasets, desc="Generating samples for datasets"):
        if model_name not in ["logistic"]:
            model = keras.models.load_model(
                os.path.join(
                    cfg.TRAINED_MODELS_DIR, dataset, f"{dataset}_{model_name}.keras"
                )
            )
        else:
            model_path = os.path.join(
                cfg.TRAINED_MODELS_DIR, dataset, f"{dataset}_{model_name}.joblib"
            )
            model = joblib.load(model_path)
        print(f"Loaded model for dataset {dataset}: {model_name}")
        if dataset not in cfg.NUMBER_OF_SAMPLES_PER_CLASS:
            X_train, X_samples, y_train, y_test = load_data(dataset, one_hot=False)
            y_samples = np.argmax(model.predict(X_samples), axis=1)

        else:
            X_samples, y_samples = get_predicted_samples(dataset, model, model_name, one_hot=False)

        file_path = os.path.join(output_dir, f"{dataset}_{model_name}_samples.csv")
        save_multivariate_ts_as_csv(file_path, X_samples, y_samples)


if __name__ == "__main__":
    create_samples("fcn")
    #create_samples("resnet")
    #create_samples("logistic")
