from sktime.transformations.panel.rocket import MiniRocketMultivariate
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

class MiniRocketLogisticModel:
    def __init__(self):
        self.transformer = MiniRocketMultivariate(random_state=42)
        self.scaler = StandardScaler(with_mean=False)
        self.classifier = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            max_iter=1000,
            multi_class="ovr",   # or "multinomial" for softmax
            solver="lbfgs"
        )

    def fit(self, X, y):
        """We are assuming that X is of shape (instances, timesteps, dimensions)"""
        X = np.transpose(X, (0, 2, 1))  # Transpose to (instances, dimensions, timesteps)
        print("Fitting MiniRocket transformer...")
        X_transformed = self.transformer.fit_transform(X)
        print("Transforming data completed.")

        print("Fitting scaler...")
        X_scaled = self.scaler.fit_transform(X_transformed)

        print("Fitting classifier...")
        self.classifier.fit(X_scaled, y)

        print("Model training completed.")

    def predict(self, X, verbose=0):
        """We are using predict_proba as predict to ensure consistency with other models"""
        X = np.transpose(X, (0, 2, 1))
        X_transformed = self.transformer.transform(X)
        X_scaled = self.scaler.transform(X_transformed)

        return self.classifier.predict_proba(X_scaled)

    def score(self, X, y):
        X = np.transpose(X, (0, 2, 1))
        X_transformed = self.transformer.transform(X)
        X_scaled = self.scaler.transform(X_transformed)

        return self.classifier.score(X_scaled, y)
