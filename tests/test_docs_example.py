"""
Verify that the docs/example.rst pipeline works end-to-end.

This script mirrors the documented example: generate synthetic data,
train a ToyFCN, compute CAM weights, generate counterfactuals, and
visualize the results.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tempfile
from pathlib import Path

from tests.toy_fcn import ToyFCN
from confetti import CONFETTI
from confetti.attribution import cam
from confetti.structs import Counterfactual
from confetti.visualizations import plot_time_series, plot_counterfactual


def generate_toy_dataset(n_samples=200, timesteps=100, channels=3, random_state=123):
    rng = np.random.default_rng(random_state)

    X = np.zeros((n_samples, timesteps, channels))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if i < n_samples // 2:
            base = np.linspace(0, 2, timesteps)
            signal = 3.0 * np.sin(np.linspace(0, 3 * np.pi, timesteps))
            noise = 0.3 * rng.normal(size=(timesteps, channels))
            X[i] = base[:, None] + signal[:, None] + noise
            y[i] = 0
        else:
            base = np.linspace(2, 0, timesteps)
            signal = 3.0 * np.sin(np.linspace(0, 12 * np.pi, timesteps))
            noise = 0.3 * rng.normal(size=(timesteps, channels))
            X[i] = base[:, None] + signal[:, None] + noise
            y[i] = 1

    return X, y


def main():
    # --- 1. Generate dataset (matches example.rst section 1) ---
    X, y = generate_toy_dataset()

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)

    assert X_train.shape == (160, 100, 3)
    assert X_test.shape == (40, 100, 3)

    # --- 2 & 3. Build and train ToyFCN (matches example.rst sections 2-3) ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = ToyFCN(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train)),
            model_path=tmp_dir,
        )
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        accuracy = (model.predict(X_test) == y_test).mean()
        print(f"Test accuracy: {accuracy:.3f}")
        assert accuracy > 0.8, f"Model accuracy too low: {accuracy}"

        # --- 4. Visualize an instance (matches example.rst section 4 intro) ---
        instance = X_test[0:1]
        plot_time_series(series=instance, title="Instance to Explain")
        print("plot_time_series succeeded")

        # --- 4 cont. Generate counterfactuals (matches example.rst section 4) ---
        weights = cam(model.model, X_train)
        print(f"CAM weights shape: {weights.shape}")

        explainer = CONFETTI(model_path=str(model.keras_file))

        results = explainer.generate_counterfactuals(
            instances_to_explain=instance,
            reference_data=X_train,
            reference_weights=weights,
            alpha=0.5,
            theta=0.51,
        )

        assert results is not None, "generate_counterfactuals returned None"

        cf_set = results[0]
        print("Original label:     ", cf_set.original_label)
        print("Counterfactual label:", cf_set.best.label)
        print("Total generated:     ", len(cf_set.all_counterfactuals))

        assert isinstance(cf_set.original_label, (int, float, np.integer, np.floating))
        assert isinstance(cf_set.best, Counterfactual)
        assert len(cf_set.all_counterfactuals) > 0
        assert cf_set.best.label != cf_set.original_label

        # --- 5. Visualize counterfactual (matches example.rst section 5) ---
        plot_counterfactual(
            original=cf_set.original_instance,
            counterfactual=cf_set.best,
            cam_weights=cf_set.feature_importance,
            cam_mode="heatmap",
            title="Counterfactual Explanation",
        )
        print("plot_counterfactual succeeded")

    print("\nAll checks passed — docs example is accurate.")


if __name__ == "__main__":
    main()
