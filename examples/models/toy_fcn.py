import keras
from keras import layers
import numpy as np
from pathlib import Path


class ToyFCN:
    """
    Minimal lightweight FCN for demonstration and example notebooks.
    Trains quickly and preserves the architectural style of a real FCN.
    """

    def __init__(self, input_shape, num_classes=2, model_dir="../examples/models"):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.model_dir / "toy_fcn.keras"

        self.model = self._build_model()

    def _build_model(self):
        """
        Build a small FCN-like architecture with drastically reduced parameters.
        """

        inputs = keras.layers.Input(shape=self.input_shape)

        # Conv block 1
        x = layers.Conv1D(filters=16, kernel_size=8, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Conv block 2
        x = layers.Conv1D(filters=32, kernel_size=5, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Conv block 3
        x = layers.Conv1D(filters=16, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Global average pooling
        gap = layers.GlobalAveragePooling1D()(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(gap)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, X_train, y_train, epochs=30, batch_size=16, verbose=0):
        """
        Train the toy FCN on small synthetic data.
        """
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2,
        )

        # Save model
        self.model.save(self.model_path)
        return self.model

    def predict(self, X):
        """
        Predict class labels.
        """
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        return self.model.predict(X)
