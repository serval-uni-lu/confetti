from typing import List, Dict
from itertools import product
import warnings
import time
from tqdm import tqdm
from pathlib import Path

from confetti import CONFETTI
from confetti.explainer.utils import load_data, load_multivariate_ts_from_csv, array_to_string
import confetti.CAM.class_activation_map as cam
from benchmark.evaluations.evaluator import Evaluator

import config as cfg

import tensorflow as tf
import pandas as pd
import keras

# TODO: Perform statistical tests and rankings at the end of the experiment


def _flag_sets_at_least_two_true() -> List[Dict[str, bool]]:
    """Return the four flag combinations with at least two objectives enabled."""
    names = ("optimize_confidence", "optimize_sparsity", "optimize_proximity")
    combos = [
        dict(zip(names, values))
        for values in product((True, False), repeat=3)
        if sum(values) >= 2
    ]
    # Order to match your examples: all three, then each pair
    priority = {
        (True, True, True): 0,
        (True, True, False): 1,
        (True, False, True): 2,
        (False, True, True): 3,
    }
    combos.sort(key=lambda d: priority[(d["optimize_confidence"], d["optimize_sparsity"], d["optimize_proximity"])])
    return combos

def _label_from_flags(flags: Dict[str, bool]) -> str:
    """Create a short label like 'conf_spar_prox' or 'conf_spar' from True flags."""
    parts = []
    if flags["optimize_confidence"]:
        parts.append("CO")
    if flags["optimize_sparsity"]:
        parts.append("SP")
    if flags["optimize_proximity"]:
        parts.append("PR")
    return "_".join(parts)

def run_objectives_experiment(model_name: str = "fcn"):
    base_dir = Path(__file__).parent
    results_dir = base_dir / "objectives_summary"
    summary = []

    for dataset in tqdm(cfg.DATASETS, desc="Datasets", unit="dataset"):
        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        model_path = str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras")

        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        model = keras.models.load_model(model_path)

        X_train, _, y_train, _ = load_data(dataset, one_hot=False)

        training_weights = cam.compute_weights_cam(model=model,
                                                   X_data=X_train,
                                                   dataset=dataset,
                                                   save_weights=False,
                                                   data_type='training')

        explainer = CONFETTI(model_path=model_path)
        evaluator = Evaluator()

        #for combo in tqdm(_flag_sets_at_least_two_true(), desc="Objective Combos", unit="combo"):
        for combo in [{'optimize_confidence': False, 'optimize_sparsity': True, 'optimize_proximity': True}]:
            label = _label_from_flags(combo)
            start_time = time.time()
            ces_naive, ces_optimized = explainer.parallelized_counterfactual_generator(
                instances_to_explain=X_samples,
                reference_data=X_train,
                reference_weights=training_weights,
                alpha=0.5,
                theta=0.51,
                proximity_distance='dtw',
                processes=8,
                verbose=False,
                **combo,
            )
            total_time = time.time() - start_time

            _, dataset_summary = evaluator.evaluate_results(
                model=model,
                explainer=label,
                dataset=dataset,
                counterfactuals=ces_optimized,
                sample=X_samples,
                og_labels=y_samples,
                training_data=X_train,
                timesteps=X_train.shape[1],
                channels=X_train.shape[2],
                alpha=True,
                param_config=0.5,
                fallback_counterfactuals=ces_naive
            )
            dataset_summary["Computation Time"] = total_time
            summary.append(dataset_summary)

            ces_naive["Solution"] = ces_naive["Solution"].apply(array_to_string)
            ces_naive.to_csv(results_dir / f"{dataset}_{label}_naive.csv", index=False)

            ces_optimized["Solution"] = ces_optimized["Solution"].apply(array_to_string)
            ces_optimized.to_csv(results_dir / f"{dataset}_{label}_optimized.csv", index=False)


    df_summary = pd.concat(summary, ignore_index=True, sort=True)
    df_summary.to_csv(results_dir / "objectives_experiment_results_SR_PR.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')
    from pymoo.config import Config
    Config.warnings['not_compiled'] = False

    run_objectives_experiment(model_name="fcn")


if __name__ == "__main__":
    main()