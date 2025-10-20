# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import keras
import tensorflow as tf
import numpy as np
import time
from pathlib import Path
from models.utils import save_logs, calculate_metrics


class ClassifierRESNET:
    def __init__(
        self,
        output_directory,
        input_shape,
        nb_classes,
        dataset_name: str,
        verbose=False,
        build=True,
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)  # ensure it exists
        self.dataset_name = dataset_name
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=8, padding="same"
        )(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, padding="same"
        )(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation("relu")(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=1, padding="same"
        )(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation("relu")(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation("relu")(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        )

        file_path = self.output_directory / f"{self.dataset_name}_resnet.keras"

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor="loss", save_best_only=True
        )

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print("error")
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        print("Training with mini_batch_size:", mini_batch_size)
        start_time = time.time()

        hist = self.model.fit(
            x_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=nb_epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=self.callbacks,
        )

        duration = time.time() - start_time
        print("Training took {} seconds.".format(duration))

        self.model.save(
            self.output_directory / f"{self.dataset_name}_last_model_resnet.keras"
        )

        y_pred = self.predict(x_val, y_true, return_df_metrics=False)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(
            self.output_directory, hist, y_pred, y_true, duration, lr=False
        )

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, return_df_metrics=True):

        model_path = self.output_directory / f"{self.dataset_name}_resnet.keras"
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
