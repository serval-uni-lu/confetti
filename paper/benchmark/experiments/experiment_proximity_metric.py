from paper import config as cfg
from pathlib import Path
import keras
import warnings
import paper.CAM.class_activation_map as cam
from src.confetti.explainer.confetti_explainer import CONFETTI
from src.confetti.utils import (
    load_data,
    load_multivariate_ts_from_csv,
    array_to_string,
)
import tensorflow as tf
import pandas as pd
from paper.benchmark.evaluations.evaluator import Evaluator
import time
from tqdm import tqdm


# TODO: Perform statistical tests and rankings at the end of the experiment


def run_proximity_metric_experiment(model_name: str = "fcn"):
    base_dir = Path(__file__).parent
    results_dir = base_dir / "proximity_metric_results"

    summary = []

    for dataset in tqdm(cfg.DATASETS, desc="Datasets", unit="dataset"):
        # Load samples and model
        sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
        model_path = str(
            cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_{model_name}.keras"
        )

        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        model = keras.models.load_model(model_path)

        # Load reference data
        X_train, _, y_train, _ = load_data(dataset, one_hot=False)

        training_weights = cam.compute_weights_cam(
            model=model,
            X_data=X_train,
            dataset=dataset,
            save_weights=False,
            data_type="training",
        )

        explainer = CONFETTI(model_path=model_path)
        evaluator = Evaluator()

        for metric in tqdm(
            ["softdtw", "ctw", "dtw", "euclidean"],
            desc="Proximity Metrics",
            unit="metric",
        ):
            start_time = time.time()
            ces_naive, ces_optimized = explainer.parallelized_counterfactual_generator(
                instances_to_explain=X_samples,
                reference_data=X_train,
                reference_weights=training_weights,
                alpha=0.5,
                theta=0.51,
                proximity_distance=metric,
                processes=8,
                verbose=False,
            )
            total_time = time.time() - start_time

            _, dataset_summary = evaluator.evaluate_results(
                model=model,
                explainer=metric,
                dataset=dataset,
                counterfactuals=ces_optimized,
                sample=X_samples,
                og_labels=y_samples,
                training_data=X_train,
                timesteps=X_train.shape[1],
                channels=X_train.shape[2],
                alpha=True,
                param_config=0.5,
                fallback_counterfactuals=ces_naive,
            )

            dataset_summary["Computation Time"] = total_time
            summary.append(dataset_summary)

            ces_naive["Solution"] = ces_naive["Solution"].apply(array_to_string)
            ces_naive.to_csv(results_dir / f"{dataset}_{metric}_naive.csv", index=False)

            ces_optimized["Solution"] = ces_optimized["Solution"].apply(array_to_string)
            ces_optimized.to_csv(
                results_dir / f"{dataset}_{metric}_optimized.csv", index=False
            )

            if metric == "dtw":
                small_window = int(X_train.shape[1] * 0.20)
                medium_window = int(X_train.shape[1] * 0.50)

                for dtw_window in tqdm(
                    [small_window, medium_window], desc="DTW Windows", unit="window"
                ):
                    start_time = time.time()

                    ces_naive, ces_optimized = (
                        explainer.parallelized_counterfactual_generator(
                            instances_to_explain=X_samples,
                            reference_data=X_train,
                            reference_weights=training_weights,
                            alpha=0.5,
                            theta=0.51,
                            proximity_distance=metric,
                            dtw_window=dtw_window,
                            processes=8,
                            verbose=False,
                        )
                    )

                    total_time = time.time() - start_time

                    _, dataset_summary = evaluator.evaluate_results(
                        model=model,
                        explainer=f"{metric}_{'small_window' if dtw_window == small_window else 'medium_window'}",
                        dataset=dataset,
                        counterfactuals=ces_optimized,
                        sample=X_samples,
                        og_labels=y_samples,
                        training_data=X_train,
                        timesteps=X_train.shape[1],
                        channels=X_train.shape[2],
                        alpha=True,
                        param_config=0.50,
                        fallback_counterfactuals=ces_naive,
                    )
                    dataset_summary["Computation Time"] = total_time

                    summary.append(dataset_summary)

                    ces_naive["Solution"] = ces_naive["Solution"].apply(array_to_string)
                    ces_naive.to_csv(
                        results_dir / f"{dataset}_{metric}_{dtw_window}_naive.csv",
                        index=False,
                    )

                    ces_optimized["Solution"] = ces_optimized["Solution"].apply(
                        array_to_string
                    )
                    ces_optimized.to_csv(
                        results_dir / f"{dataset}_{metric}_{dtw_window}_optimized.csv",
                        index=False,
                    )

    df_summary = pd.concat(summary, ignore_index=True, sort=True)
    df_summary.to_csv(results_dir / "best_proximity_metric.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    from pymoo.config import Config

    Config.warnings["not_compiled"] = False

    run_proximity_metric_experiment(model_name="fcn")


if __name__ == "__main__":
    main()
