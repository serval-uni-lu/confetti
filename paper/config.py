from pathlib import Path

# Get the path of the current script file
current_file_path = Path(__file__).resolve()

BASE_DIR = current_file_path.parent
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CAM_WEIGHTS_DIR = TRAINED_MODELS_DIR
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "benchmark" / "data"

EVALUATIONS_FILE = BASE_DIR / "benchmark" / "evaluations" / "all_evaluation_results.csv"
EXPERIMENT_PROXIMITY_METRIC = (
    BASE_DIR / "benchmark" / "experiments" / "proximity_metric_results"
)
EXPERIMENT_OBJECTIVES = BASE_DIR / "benchmark" / "experiments" / "objectives_results"
EXPERIMENT_EXECUTION_TIMES_PER_SHAPE = BASE_DIR / "benchmark" / "experiments" / "execution_times_per_shape_results"
EXPERIMENT_ABLATION_STUDY = BASE_DIR / "benchmark" / "experiments" / "ablation_study_results"

NUMBER_OF_SAMPLES_PER_CLASS = {
    "ArticularyWordRecognition": 1,
    "Epilepsy": 6,
    "ERing": 6,
    "NATOPS": 3,
    "Libras": 2,
    "RacketSports": 4,
}

# Datasets to Test
DATASETS = [
    "ArticularyWordRecognition",
    "BasicMotions",
    "Epilepsy",
    "ERing",
    "Libras",
    "RacketSports",
    "NATOPS",
]
DATASETS_SETS = [
    "ArticularyWordRecognition",
    "BasicMotions",
    "Epilepsy",
    "ERing",
    "Libras",
    "RacketSports",
]
DATASETS_TEST = ["RacketSports"]

# Fixed values for experiments
FIXED_ALPHA = 0.50
FIXED_THETA = 0.51

# default hyperparameter grids
_DEFAULT_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
_DEFAULT_THETAS = [0.55, 0.75, 0.95]

# checkpoint dict for “breaks” logic
CHECK_POINT = {
    "ArticularyWordRecognition": {
        "alphas": _DEFAULT_ALPHAS.copy(),
        "thetas": _DEFAULT_THETAS.copy(),
    },
    "BasicMotions": {
        "alphas": _DEFAULT_ALPHAS.copy(),
        "thetas": _DEFAULT_THETAS.copy(),
    },
    "Epilepsy": {"alphas": _DEFAULT_ALPHAS.copy(), "thetas": _DEFAULT_THETAS.copy()},
    "ERing": {"alphas": _DEFAULT_ALPHAS.copy(), "thetas": _DEFAULT_THETAS.copy()},
    "Libras": {"alphas": _DEFAULT_ALPHAS.copy(), "thetas": _DEFAULT_THETAS.copy()},
    "NATOPS": {"alphas": _DEFAULT_ALPHAS.copy(), "thetas": _DEFAULT_THETAS.copy()},
    "RacketSports": {
        "alphas": _DEFAULT_ALPHAS.copy(),
        "thetas": _DEFAULT_THETAS.copy(),
    },
}

ABLATION_CHECK_POINT = {
    "datasets": DATASETS.copy(),
}

TS_LENGTHS = {
    "ArticularyWordRecognition": 144,
    "BasicMotions": 100,
    "Epilepsy": 207,
    "ERing": 65,
    "Libras": 45,
    "NATOPS": 51,
    "RacketSports": 30,
}
TS_DIMENSIONS = {
    "ArticularyWordRecognition": 9,
    "BasicMotions": 6,
    "Epilepsy": 3,
    "ERing": 4,
    "Libras": 2,
    "NATOPS": 24,
    "RacketSports": 6,
}
