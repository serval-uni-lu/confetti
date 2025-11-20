import time
import json
import keras
from confetti.utils import load_data, load_multivariate_ts_from_csv
from confetti import CONFETTI
from confetti.explainer.counterfactuals import Counterfactual, CounterfactualResults
from paper import config as cfg
import pandas as pd
import numpy as np
from paper.benchmark.generators.confetti_generator import create_standard_results_dataframe
import paper.CAM.class_activation_map as cam
from paper.benchmark.evaluations.evaluator import Evaluator
from tqdm import tqdm


def experiment_ablation_study(model_name: str, ablation_study: bool = True) -> None:
    """
    Run the ablation study experiments using the confetti counterfactual generator.
    This function configures the generator with specific parameters for the ablation study
    and executes the counterfactual generation process.
    """
    # Load or initialize checkpoint.json
    checkpoint_path = cfg.EXPERIMENT_ABLATION_STUDY / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = cfg.ABLATION_CHECK_POINT.copy()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Create the execution time file
    if ablation_study:
        time_file = (cfg.EXPERIMENT_ABLATION_STUDY / f"execution_times_{model_name}_ablation.csv")
    else:
        time_file = cfg.EXPERIMENT_ABLATION_STUDY / f"execution_times_{model_name}.csv"
    if not time_file.exists():
        pd.DataFrame(columns=["Dataset", "Alpha", "Theta", "Execution Time"]).to_csv(time_file, index=False)

    print(f"Remaining datasets: {checkpoint['datasets']}")
    remaining = checkpoint["datasets"][:]
    # Run the experiment
    for dataset in tqdm(remaining, desc="Datasets"):
        print(f"Running experiment for dataset: {dataset}")
        ce_dir = cfg.EXPERIMENT_ABLATION_STUDY / dataset
        ce_dir.mkdir(parents=True, exist_ok=True)
        # Ablation Study can only be done with FCN and ResNet. Logistic doesn't use reference weights.
        model_path = str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras")
        model = keras.models.load_model(model_path)

        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)

        # load samples of the specific model
        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)

        if ablation_study is False:
            training_weights: None | np.ndarray = cam.compute_weights_cam(
                model=model,
                X_data=X_train,
                dataset=dataset,
                save_weights=True,
                data_type="training",
            )
        else:
            training_weights = None

        # Create Explainer
        ce = CONFETTI(model_path=model_path)

        start = time.time()
        counterfactuals: CounterfactualResults = ce.generate_counterfactuals(
            instances_to_explain=X_samples,
            reference_data=X_train,
            n_partitions=2,
            reference_weights=training_weights,
            processes=8,
            verbose=True,
        )
        elapsed = time.time() - start

        if ablation_study:
            output_dir = ce_dir / f"{dataset}_{model_name}_ablation_results.csv"
        else:
            output_dir = ce_dir / f"{dataset}_{model_name}_results.csv"

        counterfactuals_df = create_standard_results_dataframe(counterfactuals, X_samples, X_train)
        counterfactuals_df.to_csv(output_dir, index=False)

        # save execution time
        pd.DataFrame(
            [
                {
                    "Dataset": dataset,
                    "Execution Time": elapsed,
                }
            ]
        ).to_csv(time_file, mode="a", header=False, index=False)

        # remove dataset from checkpoint & persist
        checkpoint["datasets"].remove(dataset)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"Run finished for model={model_name}, ablation_study={ablation_study}")

def evaluate_ablation_results():
    evaluator = Evaluator()
    for model_name in tqdm(["resnet", "fcn"], desc="Models"):
        summary_evaluations = pd.DataFrame()
        exec_times_ablation = pd.read_csv(cfg.EXPERIMENT_ABLATION_STUDY / f"execution_times_{model_name}_ablation.csv")
        exec_times_normal = pd.read_csv(cfg.EXPERIMENT_ABLATION_STUDY / f"execution_times_{model_name}.csv")
        for dataset in tqdm(cfg.DATASETS, desc="Datasets"):
            for ablation in [True, False]:
                if ablation:
                    results_file = cfg.EXPERIMENT_ABLATION_STUDY / dataset / f"{dataset}_{model_name}_ablation_results.csv"
                    _, evaluation = evaluator.evaluate_from_csv(explainer="Ablation",
                                                                dataset=dataset,
                                                                model_name=model_name,
                                                                optional_path=results_file)

                    evaluation["Execution Time"] =  exec_times_ablation[exec_times_ablation["Dataset"] == dataset]["Alpha"].iloc[0]
                else:
                    results_file = cfg.EXPERIMENT_ABLATION_STUDY / dataset / f"{dataset}_{model_name}_results.csv"
                    _, evaluation  = evaluator.evaluate_from_csv(explainer="Normal",
                                                                dataset=dataset,
                                                                model_name=model_name,
                                                                optional_path=results_file)
                    evaluation["Execution Time"] =  exec_times_normal[exec_times_normal["Dataset"] == dataset]["Alpha"].iloc[0]
                summary_evaluations = pd.concat([summary_evaluations, evaluation], ignore_index=True)
        summary_path = cfg.EXPERIMENT_ABLATION_STUDY / f"ablation_summary_{model_name}.csv"
        summary_evaluations.to_csv(summary_path, index=False)


def main():
    #experiment_ablation_study("resnet", ablation_study=True)
    #experiment_ablation_study("fcn", ablation_study=True)
    #experiment_ablation_study("resnet", ablation_study=False)
    #experiment_ablation_study("fcn", ablation_study=True)
    evaluate_ablation_results()



if __name__ == "__main__":
    main()


