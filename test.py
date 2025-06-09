import pandas as pd

import config as cfg
import numpy as np
import keras
from sklearn import preprocessing
from pathlib import Path
import warnings
import tensorflow as tf
from confetti.CAM import class_activation_map as cam
tf.keras.utils.disable_interactive_logging()
# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning)
from confetti import CONFETTI
from benchmark import Evaluator
from confetti.explainer.utils import load_multivariate_ts_from_csv
from sktime.datasets import load_UCR_UEA_dataset

if __name__ == "__main__":
    dataset = 'ArticularyWordRecognition'

    #Data will load with shape (instances, dimensions, timesteps)
    X_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_type="numpy3d")
    # Reshape data to (instances, timesteps, dimensions)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    # Encode
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    #Load instances to explain
    sample_path = f"{cfg.DATA_DIR}/{dataset}_samples.csv"
    X_samples, y_samples = load_multivariate_ts_from_csv(sample_path)


    input_shape = X_train.shape[1:] #The input shape for our CNN should be (timesteps, dimensions)

    model_path = cfg.TRAINED_MODELS_DIR / dataset / f'{dataset}_fcn.keras'
    model = keras.models.load_model(str(model_path))

    weights = np.load(f"/Users/alan.paredes/Desktop/confetti/models/trained_models/{dataset}/{dataset}_training_weights.npy")



    ce = CONFETTI(model_path, X_train, X_samples, y_samples, y_train, weights)
    thetas = [0.65, 0.75,0.85, 0.95]
    summaries = pd.DataFrame(columns=['Theta', 'Dataset', 'Coverage', 'Sparsity', 'Confidence', 'Validity', 'Proximity'])
    for theta in thetas:
        naive, optimized = ce.parallelized_counterfactual_generator(save_counterfactuals=False, processes=5, theta=theta)

        ev = Evaluator()

        metrics, summary = ev.evaluate_results(model,
                                               dataset=dataset,
                                               counterfactuals=optimized,
                                               sample=X_samples,
                                               og_labels=y_samples,
                                               timesteps=input_shape[0],
                                               channels=input_shape[1])
        summary['Theta'] = theta
        summaries = pd.concat([summaries, summary], ignore_index=True)

    summaries.to_csv("summary.csv", index=False)
