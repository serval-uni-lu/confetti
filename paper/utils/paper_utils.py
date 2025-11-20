from paper import config as cfg
from confetti.utils import load_data
import numpy as np
from typing import Tuple, List, Any

def get_samples(dataset, one_hot=False):
    X_train, X_test, y_train, y_test = load_data(
        dataset, one_hot=one_hot
    )  # Pass one_hot flag

    # Get available classes (based on non-one-hot labels)
    available_classes = np.unique(np.argmax(y_train, axis=1) if one_hot else y_train)

    # Lists to store selected samples
    X_test_samples = []
    y_test_samples = []

    for cls in available_classes:
        # Find indices of the samples belonging to the class
        if one_hot:
            test_indices = np.where(np.argmax(y_test, axis=1) == cls)[0]
        else:
            test_indices = np.where(y_test == cls)[0]

        if test_indices.size > 0:
            number_samples = cfg.NUMBER_OF_SAMPLES_PER_CLASS[dataset]
            selected_test_indices = np.random.choice(
                test_indices, size=number_samples, replace=False
            )

            X_test_samples.extend(X_test[selected_test_indices])
            y_test_samples.extend(y_test[selected_test_indices])

    # Convert lists back to numpy arrays
    X_test_samples = np.array(X_test_samples)
    y_test_samples = np.array(y_test_samples)

    return X_test_samples, y_test_samples

def get_predicted_samples(
    dataset: str,
    model: Any,
    model_name: str,
    one_hot: bool = False,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a test-set sample of size
        len(available_classes) * cfg.NUMBER_OF_SAMPLES_PER_CLASS[dataset]

    Selection is driven by *predicted* labels.  If some classes have
    fewer predictions than requested, the deficit is filled by drawing
    extra items from the other classes (one-by-one, round-robin)
    until the overall target size is met.

    Parameters
    ----------
    dataset : str
    model   : keras.Model
    one_hot : bool, default False
    rng     : np.random.Generator, optional
              Allows reproducible sampling if you pass e.g. np.random.default_rng(42).

    Returns
    -------
    X_sel : np.ndarray
    y_pred_sel : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng(42)  # modern RNG

    # ── 1. Load data ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, _ = load_data(dataset, one_hot=one_hot)

    # ── 2. Predict labels ──────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test)
    y_pred = (
        np.argmax(y_pred_prob, axis=1)
        if y_pred_prob.ndim > 1
        else y_pred_prob.astype(int)
    )  # binary models

    # ── 3. Build per-class index pools ─────────────────────────────────────────
    classes: np.ndarray = np.unique(y_pred)

    pools: dict[int, List[int]] = {
        cls: rng.permutation(np.where(y_pred == cls)[0]).tolist() for cls in classes
    }

    n_per_cls = cfg.NUMBER_OF_SAMPLES_PER_CLASS[dataset]
    target_len = len(classes) * n_per_cls

    selected_indices: list[int] = []

    # ── 4. First pass: take up to n_per_cls from each class ────────────────────
    for cls in classes:
        take = min(n_per_cls, len(pools[cls]))
        selected_indices.extend(pools[cls][:take])
        pools[cls] = pools[cls][take:]  # remove taken

    # ── 5. Second pass: round-robin top-up across remaining pools ──────────────
    # Keep cycling through classes and grabbing one at a time
    cls_cycle = list(classes)  # deterministic order
    cycle_idx = 0
    while len(selected_indices) < target_len:
        cls = cls_cycle[cycle_idx % len(cls_cycle)]
        if pools[cls]:  # still something left?
            selected_indices.append(pools[cls].pop(0))
        # Break if no class has anything left
        if all(len(p) == 0 for p in pools.values()):
            break
        cycle_idx += 1

    if len(selected_indices) < target_len:
        # Optional: log/print a warning in your own logger
        print(
            f"[get_predicted_samples] Warning: could only gather "
            f"{len(selected_indices)} / {target_len} samples "
            f"because X_test is exhausted."
        )

    # ── 6. Final arrays ────────────────────────────────────────────────────────
    selected_indices = np.asarray(selected_indices, dtype=int)
    X_samples = X_test[selected_indices]
    y_pred_samples = y_pred[selected_indices]

    return X_samples, y_pred_samples