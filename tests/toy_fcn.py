import keras
from keras import layers
import numpy as np
from pathlib import Path

class ToyFCN:

    def __init__(self, input_shape, num_classes= 2, model_path= None):
        self.input_shape   = input_shape
        self.num_classes   = num_classes
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model         = self._build_model()

    @property
    def keras_file(self) -> Path:
        """Full path to the saved Keras model file."""
        return self.model_path / "toy_fcn.keras"

    def _build_model(self):
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

    def fit(self, X_train, y_train, epochs=20, batch_size=16, verbose=1):
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2,
        )
        self.model.save(self.keras_file)
        return self.model

    def predict(self, X):
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)