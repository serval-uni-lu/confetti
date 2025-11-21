import numpy as np
from confetti.utils import save_multivariate_ts_as_csv
from pathlib import Path

def generate_toy_dataset(
    n_samples=200, timesteps=100, channels=3, random_state=123
):
    """
    Generate a synthetic multivariate time series dataset with strong,
    class-distinguishing structure so that a small FCN can reach >80% accuracy.
    """

    rng = np.random.default_rng(random_state)

    X = np.zeros((n_samples, timesteps, channels))
    y = np.zeros(n_samples, dtype=int)

    # Define class patterns
    for i in range(n_samples):
        if i < n_samples // 2:
            # Class 0
            base = np.linspace(0, 2, timesteps)                        # rising trend
            signal = 3.0 * np.sin(np.linspace(0, 3 * np.pi, timesteps)) # low freq
            noise = 0.3 * rng.normal(size=(timesteps, channels))
            X[i] = (base[:, None] + signal[:, None]) + noise
            y[i] = 0
        else:
            # Class 1
            base = np.linspace(2, 0, timesteps)                         # falling trend
            signal = 3.0 * np.sin(np.linspace(0, 12 * np.pi, timesteps)) # high freq
            noise = 0.3 * rng.normal(size=(timesteps, channels))
            X[i] = (base[:, None] + signal[:, None]) + noise
            y[i] = 1

    return X, y


def main():
    X, y = generate_toy_dataset()

    # Shuffle before split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Save datasets
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "examples" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_multivariate_ts_as_csv(DATA_DIR / "toy_train.csv", X_train, y_train)
    save_multivariate_ts_as_csv(DATA_DIR / "toy_test.csv", X_test, y_test)
    print(f"Saved toy dataset to {DATA_DIR}")


if __name__ == "__main__":
    main()
