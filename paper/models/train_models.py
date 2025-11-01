from paper.models.resnet import ClassifierRESNET
from paper.models.fcn import ClassifierFCN
from src.confetti.utils import load_data
import numpy as np
from paper import config as cfg
import keras


def train_model(model_name: str):
    DATASETS = cfg.DATASETS
    for dataset in DATASETS:
        output_dir = cfg.TRAINED_MODELS_DIR / dataset

        # Data will load with shape (instances, dimensions, timesteps)
        # We are not one-hot encoding the labels here, as the later the metrics expect labels to be in integer format.
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
        print("Data loaded successfully.")

        # We encode here the labels.
        y_train_encoded, y_test_encoded = (
            keras.utils.to_categorical(y_train),
            keras.utils.to_categorical(y_test),
        )
        print("Labels encoded successfully.")

        nb_classes = len(np.unique(np.concatenate([y_train, y_test])))
        input_shape = X_train.shape[
            1:
        ]  # The input shape for our DNN should be (timesteps, dimensions)

        if model_name == "fcn":
            print("Creating FCN model...")
            model = ClassifierFCN(
                input_shape=input_shape,
                nb_classes=nb_classes,
                dataset_name=dataset,
                verbose=True,
            )
        else:
            print("Creating ResNet model...")
            model = ClassifierRESNET(
                output_directory=output_dir,
                input_shape=input_shape,
                nb_classes=nb_classes,
                dataset_name=dataset,
                verbose=True,
            )
        print("Model created successfully.")
        print("Training starting...")
        model.fit(
            x_train=X_train,
            y_train=y_train_encoded,
            x_val=X_test,
            y_val=y_test_encoded,
            y_true=y_test,
        )  # y_true is used for metrics calculation


def main():
    # This script will train FCN and ResNet models for all datasets.
    train_model("fcn")
    train_model("resnet")


if __name__ == "__main__":
    main()
