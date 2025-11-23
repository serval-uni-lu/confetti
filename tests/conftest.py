import pytest
import keras
import numpy as np

from tests.toy_fcn import ToyFCN
from confetti.explainer.explainer import CONFETTI
from confetti.attribution import cam


@pytest.fixture(scope="session")
def toy_dataset():
    """
    Realistic toy dataset used for CONFETTI unit tests.
    Produces structured MTS with class-distinguishing patterns.
    """

    def generate_toy_dataset(
        n_samples=200, timesteps=100, channels=3, random_state=123
    ):
        rng = np.random.default_rng(random_state)
        X = np.zeros((n_samples, timesteps, channels))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            if i < n_samples // 2:
                # Class 0
                base = np.linspace(0, 2, timesteps)                         # rising trend
                signal = 3.0 * np.sin(np.linspace(0, 3*np.pi, timesteps))   # low freq
                noise = 0.3 * rng.normal(size=(timesteps, channels))
                X[i] = base[:, None] + signal[:, None] + noise
                y[i] = 0
            else:
                # Class 1
                base = np.linspace(2, 0, timesteps)                         # falling trend
                signal = 3.0 * np.sin(np.linspace(0, 12*np.pi, timesteps))  # high freq
                noise = 0.3 * rng.normal(size=(timesteps, channels))
                X[i] = base[:, None] + signal[:, None] + noise
                y[i] = 1

        return X, y

    # Generate dataset
    X, y = generate_toy_dataset()

    # Shuffle before split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 80/20 split
    split = int(0.8 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test

@pytest.fixture(scope="session")
def toy_model_path(tmp_path_factory, toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset

    tmp_dir = tmp_path_factory.mktemp("toy_model_dir")
    model = ToyFCN(
        input_shape=X_train.shape[1:],
        num_classes=2,
        model_path=tmp_dir
    )

    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    return model.keras_file

@pytest.fixture(scope="session")
def cam_weights(toy_model_path, toy_dataset):
    """
    CAM Weights from Training Sample
    """

    X_train, y_train, X_test, y_test = toy_dataset

    model = keras.models.load_model(toy_model_path)

    weights = cam(model, X_train)

    return weights

@pytest.fixture(scope="session")
def nun_indices(instances_to_explain, reference_data, reference_labels, confetti_instance):
    """
    For each instance to explain, select exactly ONE NUN from the reference dataset.

    NUN is defined as any reference sample that has a different predicted label
    from the test instance. This avoids calling _nearest_unlike_neighbour
    inside tests and guarantees stable, deterministic NUNs.

    Returns
    -------
    list[int]
        A list of length len(instances_to_explain), where each element
        is the selected NUN index for that instance.
    """

    conf = confetti_instance
    test_labels = conf.original_labels  # predicted labels for instances_to_explain

    nun_list = []

    for i, test_label in enumerate(test_labels):

        # find all reference indices with opposite class
        opposite_indices = np.where(reference_labels != test_label)[0]

        if opposite_indices.size == 0:
            raise RuntimeError(
                f"Could not find an unlike-class neighbour for instance {i}. "
                f"Your toy dataset needs both classes represented in reference_data."
            )

        # Deterministically pick the first opposite-class reference point
        nun_list.append(int(opposite_indices[0]))

    return nun_list

@pytest.fixture()
def instances_to_explain(toy_dataset):
    _, _, X_test, _ = toy_dataset
    return X_test[:5]

@pytest.fixture()
def reference_data(toy_dataset):
    X_train, _, _, _ = toy_dataset
    return X_train

@pytest.fixture()
def reference_labels(reference_data, toy_model_path):
    model = keras.models.load_model(toy_model_path)
    return np.argmax(model.predict(reference_data), axis=1)

@pytest.fixture()
def confetti_instance(
    toy_model_path,
    instances_to_explain,
    reference_data,
    reference_labels,
    cam_weights
):
    """Fully configured CONFETTI object."""

    conf = CONFETTI(model_path=toy_model_path)

    conf.instances_to_explain = instances_to_explain
    conf.original_labels = np.argmax(conf.model.predict(instances_to_explain), axis=1)

    conf.reference_data = reference_data
    conf.reference_labels = reference_labels
    conf.weights = cam_weights

    return conf
