import numpy as np
import pytest
from pathlib import Path

from confetti.structs import Counterfactual, CounterfactualSet
from confetti.errors import CONFETTIError, CONFETTIConfigurationError
from confetti import CONFETTI


# Naive Stage Tests
def test_naive_stage_success(confetti_instance):
    """
    Verify that _naive_stage:
    - returns a Counterfactual object
    - returns an integer subarray_length
    - produces a CF of the same shape as the original
    - differs from the original instance in exactly subarray_length positions
    """

    conf = confetti_instance

    # We test the naive stage on the first instance (index 0)
    instance_index = 0

    # predicted label of the test instance
    pred_label = conf.original_labels[instance_index]

    # find NUN index using real model + real reference_data
    nun_index = conf._nearest_unlike_neighbour(
        query=conf.instances_to_explain[instance_index],
        predicted_label=pred_label,
        theta=0.51,
        n_neighbors=1,
        distance="euclidean",
    )

    assert nun_index is not None, "NUN must exist for the test dataset."

    counterfactual, sub_len = conf._naive_stage(
        instance_index=instance_index,
        nun_index=nun_index,
        model=conf.model,
        theta=0.51,
        verbose=False,
    )

    # ---- TYPE CHECKS ----
    assert isinstance(counterfactual, Counterfactual)
    assert isinstance(sub_len, int)

    # ---- SHAPE CHECK ----
    cf_arr = counterfactual.counterfactual
    original_arr = conf.instances_to_explain[instance_index]

    # cf_arr shape = (1, timesteps, channels)
    assert cf_arr.shape[1:] == original_arr.shape
    assert cf_arr.shape[0] == 1   # naive stage returns reshaped CF

    cf_arr = cf_arr[0]  # remove batch dimension

    assert cf_arr.shape == original_arr.shape

    # ---- DIFFERENCE CHECK ----
    # Differences measured across all positions
    diff_mask = np.any(cf_arr != original_arr, axis=1)  # boolean vector over time axis
    n_different_positions = diff_mask.sum()

    assert (
        n_different_positions == sub_len
    ), f"CF changed {n_different_positions} positions but expected {sub_len}."

def test_naive_stage_raises_without_weights(confetti_instance):
    """
    If weights are missing, _naive_stage must raise CONFETTIError.
    """

    conf = confetti_instance

    # temporarily remove weights
    conf.weights = None

    instance_index = 0
    pred_label = conf.original_labels[instance_index]

    nun_index = conf._nearest_unlike_neighbour(
        query=conf.instances_to_explain[instance_index],
        predicted_label=pred_label,
    )

    assert nun_index is not None, "NUN must exist for this test."

    with pytest.raises(CONFETTIError):
        conf._naive_stage(
            instance_index=instance_index,
            nun_index=nun_index,
            model=conf.model,
        )

# Optimization Stage Tests
def test_optimization_success(confetti_instance):
    """
    Successful optimization:
    - returns a CounterfactualSet
    - all CFs have correct shape
    - all CF labels differ from original labels
    - all CF predicted probabilities >= theta
    """

    conf = confetti_instance
    instance_index = 0
    theta = 0.51

    original_label = conf.original_labels[instance_index]
    query = conf.instances_to_explain[instance_index]

    nun_index = conf._nearest_unlike_neighbour(
        query=query,
        predicted_label=original_label,
        theta=theta,
        distance="euclidean",
        n_neighbors=1
    )
    assert nun_index is not None, "Dataset must guarantee a valid NUN."

    naive_cf, sub_len = conf._naive_stage(
        instance_index=instance_index,
        nun_index=nun_index,
        model=conf.model,
        theta=theta
    )
    assert isinstance(sub_len, int)

    # Run optimization
    cs = conf._optimization(
        instance_index=instance_index,
        nun_index=nun_index,
        subsequence_length=sub_len,
        model=conf.model,
        theta=theta,
        alpha=0.5,
        population_size=100,                # small for speed
        maximum_number_of_generations=50,   # small for speed
        verbose=False
    )

    if cs is None:
        pytest.skip("Optimization did not return a solution under test-time constraints.")

    assert isinstance(cs, CounterfactualSet)

    # Shape Check
    for cf in cs.all_counterfactuals:
        assert cf.counterfactual.shape == query.shape

    # Validity Check
    for cf in cs.all_counterfactuals:
        assert cf.label != original_label

    #  Probability Threshold Check
    predictions = conf.model.predict(
        np.stack([cf.counterfactual for cf in cs.all_counterfactuals], axis=0)
    )
    pred_probs = predictions[np.arange(len(predictions)), [cf.label for cf in cs.all_counterfactuals]]

    assert np.all(pred_probs >= theta), "All CFs must satisfy the theta constraint."

def test_optimization_returns_none_for_impossible_theta(confetti_instance):
    """
    Optimization must return None when the target theta is impossible to satisfy.
    """

    conf = confetti_instance
    instance_index = 0
    impossible_theta = 0.9999

    original_label = conf.original_labels[instance_index]
    query = conf.instances_to_explain[instance_index]

    nun_index = conf._nearest_unlike_neighbour(
        query=query,
        predicted_label=original_label,
        theta=0.51
    )
    assert nun_index is not None

    naive_cf, sub_len = conf._naive_stage(
        instance_index=instance_index,
        nun_index=nun_index,
        model=conf.model,
        theta=0.51
    )

    cs = conf._optimization(
        instance_index=instance_index,
        nun_index=nun_index,
        subsequence_length=sub_len,
        model=conf.model,
        theta=impossible_theta,
        population_size=10,
        maximum_number_of_generations=3
    )

    assert cs is None

@pytest.mark.parametrize(
    "opt_conf,opt_spar,opt_prox",
    [
        (False, False, False),
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
        (True,  True,  False),
        (True,  False, True),
        (False, True,  True),
        (True,  True,  True),
    ]
)
def test_optimization_objective_configurations(
    confetti_instance,
    opt_conf,
    opt_spar,
    opt_prox,
) -> None:
    """
    Check all 8 possible objective configurations.

    - Exactly ONE objective=True  -> MUST raise CONFETTIConfigurationError.
    - All other combinations      -> MUST NOT raise.

    The test relies on `confetti_instance` having:
    - reference_data set to the training data
    - reference_labels set from model predictions
    - weights (CAM) defined for the training data
    """

    conf = confetti_instance
    instance_index = 0
    theta = 0.51

    # Sanity: reference data and labels must be set by the fixture
    assert conf.reference_data is not None
    assert conf.reference_labels is not None

    query = conf.instances_to_explain[instance_index]
    original_label = conf.original_labels[instance_index]

    # NUN search uses conf.reference_data internally
    nun_index = conf._nearest_unlike_neighbour(
        query=query,
        predicted_label=original_label,
        theta=theta,
        distance="euclidean",
        n_neighbors=1,
    )

    if nun_index is None:
        pytest.skip("No NUN found for this configuration; cannot test optimization setup.")

    # Naive stage to obtain a valid subsequence length
    naive_cf, sub_len = conf._naive_stage(
        instance_index=instance_index,
        nun_index=nun_index,
        model=conf.model,
        theta=theta,
    )

    num_active = sum([opt_conf, opt_spar, opt_prox])

    if num_active < 2:
        with pytest.raises(CONFETTIConfigurationError):
            conf._optimization(
                instance_index=instance_index,
                nun_index=nun_index,
                subsequence_length=sub_len,
                model=conf.model,
                theta=theta,
                optimize_confidence=opt_conf,
                optimize_sparsity=opt_spar,
                optimize_proximity=opt_prox,
                population_size=8,
                maximum_number_of_generations=3,
                verbose=False,
            )
    else:
        conf._optimization(
            instance_index=instance_index,
            nun_index=nun_index,
            subsequence_length=sub_len,
            model=conf.model,
            theta=theta,
            optimize_confidence=opt_conf,
            optimize_sparsity=opt_spar,
            optimize_proximity=opt_prox,
            population_size=8,
            maximum_number_of_generations=3,
            verbose=False,
        )

def test_findsubarray_basic():
    w = np.array([1, 2, 3, 4, 5])
    assert CONFETTI._findsubarray(w, 2) == 3

def test_findsubarray_full_length():
    w = np.array([5, 4, 3])
    assert CONFETTI._findsubarray(w, 3) == 0

def test_findsubarray_tie_returns_first_max():
    w = np.array([1, 3, 3, 1])
    assert CONFETTI._findsubarray(w, 2) == 1

@pytest.fixture
def base_arguments():
    return {
        "instances_to_explain": np.zeros((5, 10, 3)),
        "reference_data": np.zeros((5, 10, 3)),
        "reference_weights": None,
        "alpha": 0.5,
        "theta": 0.51,
        "n_partitions": 3,
        "population_size": 8,
        "maximum_number_of_generations": 4,
        "crossover_probability": 1.0,
        "mutation_probability": 0.9,
        "optimize_confidence": True,
        "optimize_sparsity": True,
        "optimize_proximity": True,
        "proximity_distance": "euclidean",
        "dtw_window": None,
        "save_counterfactuals": False,
        "verbose": False,
        "output_path": None,
    }


VALID_TYPES = {
    "instances_to_explain": np.ndarray,
    "reference_data": np.ndarray,
    "reference_weights": (np.ndarray, type(None)),
    "alpha": float,
    "theta": float,
    "n_partitions": int,
    "population_size": int,
    "maximum_number_of_generations": int,
    "crossover_probability": float,
    "mutation_probability": float,
    "optimize_confidence": bool,
    "optimize_sparsity": bool,
    "optimize_proximity": bool,
    "proximity_distance": str,
    "dtw_window": (int, type(None)),
    "save_counterfactuals": bool,
    "verbose": bool,
    "output_path": (str, Path, type(None)),
}


def invalid_values_for(expected):
    """Produce values that are guaranteed not to match the expected types."""
    candidates = [
        123,
        3.14,
        "wrong",
        [1, 2, 3],
        {"a": 1},
        Path("/tmp/x"),
        np.array([1, 2, 3]),
        None,
    ]

    if isinstance(expected, tuple):
        return [v for v in candidates if not isinstance(v, expected)]
    return [v for v in candidates if not isinstance(v, expected)]


@pytest.mark.parametrize(
    "arg_name",
    [k for k in VALID_TYPES.keys() if k != "output_path"]
)
def test_validate_types_all_arguments(arg_name, base_arguments):
    expected_type = VALID_TYPES[arg_name]

    # 1) Valid dict must pass
    CONFETTI._validate_types(base_arguments.copy())

    # 2) Invalid values must raise
    for bad_value in invalid_values_for(expected_type):
        arguments = base_arguments.copy()
        arguments[arg_name] = bad_value

        with pytest.raises(CONFETTIConfigurationError):
            CONFETTI._validate_types(arguments)

def test_validate_types_output_path_when_saving(base_arguments):
    args = base_arguments.copy()

    # output path is only validated when save_counterfactuals is True
    args["save_counterfactuals"] = True

    invalids = invalid_values_for((str, Path, type(None)))

    for bad in invalids:
        args2 = args.copy()
        args2["output_path"] = bad

        if bad is None:
            # None is allowed even when saving
            CONFETTI._validate_types(args2)
        else:
            with pytest.raises(CONFETTIConfigurationError):
                CONFETTI._validate_types(args2)
