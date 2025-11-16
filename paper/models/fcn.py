import keras
import numpy as np
from keras.optimizers import Adam
from pathlib import Path


class ClassifierFCN:
    def __init__(
        self, input_shape, nb_classes, dataset_name, verbose=False, build=True
    ):
        # Define the base directory relative to the current script's location
        self.model = None
        self.base_output_directory = (
            Path(__file__).resolve().parent / "trained_models" / dataset_name
        )
        self.base_output_directory.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist

        self.dataset_name = dataset_name

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(shape=input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(
            input_layer
        )
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(gap_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=Adam(),
            metrics=["accuracy"],
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        )

        # Save only the best model during training
        file_path = self.base_output_directory / f"{self.dataset_name}_fcn.keras"
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(file_path), monitor="loss", save_best_only=True
        )

        self.callbacks = [reduce_lr, model_checkpoint]
        self.model = model

        return model

    def fit(self, x_train, y_train, x_test, y_test):
        batch_size = 16
        nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        self.model.fit(
            x_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=(x_test, y_test),
        )

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def predict_proba(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred
