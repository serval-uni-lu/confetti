import numpy as np

from paper import config as cfg
import keras
import warnings
import paper.CAM.class_activation_map as cam
from confetti import CONFETTI
from confetti.explainer.counterfactuals import CounterfactualResults
from confetti.utils import load_data, load_multivariate_ts_from_csv, array_to_string
import pandas as pd
from tqdm import tqdm
import time
import json

warnings.filterwarnings("ignore", category=FutureWarning)
keras.utils.disable_interactive_logging()

def create_standard_results_dataframe(
    counterfactuals: CounterfactualResults,
    test_instances: np.ndarray,
    nun_instances: np.ndarray
) -> pd.DataFrame:
    """Create the standard DataFrame used by the Evaluator from CounterfactualResults."""
    df = counterfactuals.to_dataframe()
    df["counterfactual"] = df["counterfactual"].apply(array_to_string)

    df["Test Instance"] = [
        next((i for i, instance in enumerate(test_instances) if np.array_equal(original, instance)), None)
        for original in df["original_instance"]
    ]

    df["NUN Instance"] = [
        next((i for i, instance in enumerate(nun_instances) if np.array_equal(nun, instance)), None)
        for nun in df["nearest_unlike_neighbour"]
    ]

    df = df.rename(columns={"label": "CE Label", "counterfactual": "Solution"})
    return df[["Solution", "Test Instance", "NUN Instance", "CE Label"]].reset_index(drop=True)


def run_confetti_counterfactuals(model_name=None, ablation_study=False) -> None:
    """
    Run CONFETTI counterfactual experiments for given model and datasets.

    Arguments
    ---------
    model_name : str
        Name of the trained model to use (e.g., 'resnet', 'fcn').
    ablation_study : bool
        Whether to perform an ablation study (without naive stage).
    """

    # Load or initialize checkpoint.json
    checkpoint_path = cfg.RESULTS_DIR / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = cfg.CHECK_POINT.copy()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Create the execution time file
    if ablation_study:
        time_file = (cfg.RESULTS_DIR / f"execution_times_confetti_{model_name}_ablation_study.csv")
    else:
        time_file = cfg.RESULTS_DIR / f"execution_times_confetti_{model_name}.csv"
    if not time_file.exists():
        pd.DataFrame(columns=["Dataset", "Alpha", "Theta", "Execution Time"]).to_csv(time_file, index=False)

    # Run the experiment
    for dataset, params in checkpoint.items():
        ce_dir = cfg.RESULTS_DIR / dataset
        ce_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras")

        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
        model = keras.models.load_model(model_path)

        # load samples of the specific model
        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        if ablation_study is False:
            training_weights : np.ndarray = cam.compute_weights_cam(
                model=model,
                X_data=X_train,
                dataset=dataset,
                save_weights=True,
                data_type="training",
            )

        # Create Explainer
        ce = CONFETTI(model_path=model_path)

        # — Experiment 1: Variate Alphas —
        alphas = params.get("alphas", [])
        for alpha in tqdm(
            alphas.copy(),
            desc=f"{dataset} ⟶ alphas ({len(alphas)} remaining)",
            leave=False,
        ):
            start = time.time()
            if ablation_study:
                # To perform ablation of naive stage, simply do not pass training weights
                counterfactuals : CounterfactualResults = ce.generate_counterfactuals(
                    instances_to_explain=X_samples,
                    reference_data=X_train,
                    alpha=alpha,
                    theta=cfg.FIXED_THETA,
                    n_partitions=2,
                    processes=8,
                    verbose=True,
                )
            else:
                counterfactuals : CounterfactualResults  = ce.generate_counterfactuals(
                    instances_to_explain=X_samples,
                    reference_data=X_train,
                    reference_weights=training_weights,
                    alpha=alpha,
                    theta=cfg.FIXED_THETA,
                    n_partitions=2,
                    processes=8,
                    verbose=True,
                )
            elapsed = time.time() - start


            if ablation_study:
                output_dir = ce_dir / f"confetti_{dataset}_{model_name}_alpha_{alpha}_ablation_study.csv"
            else:
                output_dir = ce_dir / f"confetti_{dataset}_{model_name}_alpha_{alpha}.csv"

            counterfactuals_df = create_standard_results_dataframe(counterfactuals, X_samples, X_train)
            counterfactuals_df.to_csv(output_dir, index=False)


            # save execution time
            pd.DataFrame(
                [
                    {
                        "Dataset": dataset,
                        "Alpha": alpha,
                        "Theta": cfg.FIXED_THETA,
                        "Execution Time": elapsed,
                    }
                ]
            ).to_csv(time_file, mode="a", header=False, index=False)

            # remove from checkpoint & persist
            params["alphas"].pop(0)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        # — Experiment 2: Variate Thetas —
        thetas = params.get("thetas", [])
        for theta in tqdm(
            thetas.copy(),
            desc=f"{dataset} ⟶ thetas ({len(thetas)} remaining)",
            leave=False,
        ):
            start = time.time()
            if ablation_study:
                # To perform ablation of naive stage, simply do not pass training weights
                counterfactuals : CounterfactualResults = ce.generate_counterfactuals(
                    instances_to_explain=X_samples,
                    reference_data=X_train,
                    alpha=cfg.FIXED_ALPHA,
                    theta=theta,
                    n_partitions=2,
                    processes=8,
                    verbose=True,
                )
            else:
                # Use training weights if not in ablation study
                counterfactuals : CounterfactualResults = ce.generate_counterfactuals(
                    instances_to_explain=X_samples,
                    reference_data=X_train,
                    reference_weights=training_weights,
                    alpha=cfg.FIXED_ALPHA,
                    theta=theta,
                    n_partitions=2,
                    processes=8,
                    verbose=True,
                )
            elapsed = time.time() - start

            if ablation_study:
                output_dir = ce_dir / f"confetti_{dataset}_{model_name}_theta_{theta}_ablation_study.csv"
            else:
                output_dir = ce_dir / f"confetti_{dataset}_{model_name}_theta_{theta}.csv"

            counterfactuals_df = create_standard_results_dataframe(counterfactuals, X_samples, X_train)
            counterfactuals_df.to_csv(output_dir, index=False)

            pd.DataFrame(
                [
                    {
                        "Dataset": dataset,
                        "Alpha": cfg.FIXED_ALPHA,
                        "Theta": theta,
                        "Execution Time": elapsed,
                    }
                ]
            ).to_csv(time_file, mode="a", header=False, index=False)

            # remove from checkpoint & persist
            params["thetas"].pop(0)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)


def main():
    run_confetti_counterfactuals(model_name="resnet", ablation_study=False)


if __name__ == "__main__":
    main()
