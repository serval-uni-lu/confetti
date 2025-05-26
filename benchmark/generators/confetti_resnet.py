import config as cfg
import keras
from pathlib import Path
import warnings
import confetti.CAM.class_activation_map as cam
from confetti.explainer.confetti_explainer import CONFETTI
from confetti.explainer.utils import load_data, load_multivariate_ts_from_csv, array_to_string
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import time
import json

tf.keras.utils.disable_interactive_logging()
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    # — load or initialize checkpoint.json —
    cp_path = cfg.RESULTS_DIR / "checkpoint.json"
    if cp_path.exists():
        with open(cp_path) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = cfg.CHECK_POINT.copy()
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    time_file = cfg.RESULTS_DIR / "execution_times_confetti_resnet.csv"
    if not time_file.exists():
        pd.DataFrame(columns=['Dataset', 'Alpha', 'Theta', 'Execution Time']) \
            .to_csv(time_file, index=False)

    for dataset, params in checkpoint.items():
        ce_dir = cfg.RESULTS_DIR / dataset
        ce_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_resnet.keras")

        # load data, model, explainer
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
        model = keras.models.load_model(model_path)

        # load samples
        sample_file = f"{cfg.DATA_DIR}/{dataset}_samples.csv"
        X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)
        training_weights = cam.compute_weights_cam(model=model, X_data=X_train, dataset=dataset,
                                                   save_weights=True, data_type='training')

        #Create Explainer
        ce = CONFETTI(model_path, X_train, X_samples, y_samples, y_train, training_weights)

        # — Experiment 1: Alphas —
        alphas = params.get('alphas', [])
        for alpha in tqdm(alphas.copy(),
                          desc=f"{dataset} ⟶ alphas ({len(alphas)} remaining)",
                          leave=False):
            start = time.time()
            ces_naive, ces_optimized = ce.parallelized_counterfactual_generator(
                ce_dir,
                save_counterfactuals=False,
                processes=6,
                alpha=alpha,
                theta=cfg.FIXED_THETA
            )
            elapsed = time.time() - start

            # Convert Solution arrays to space-separated strings for CSV compatibility
            ces_naive["Solution"] = ces_naive["Solution"].apply(array_to_string)
            ces_optimized["Solution"] = ces_optimized["Solution"].apply(array_to_string)

            # save CFs + timing
            ces_naive.to_csv(ce_dir / f"confetti_naive_{dataset}_resnet_alpha_{alpha}.csv", index=False)
            ces_optimized.to_csv(ce_dir / f"confetti_optimized_{dataset}_resnet_alpha_{alpha}.csv", index=False)
            # save execution time
            pd.DataFrame([{
                'Dataset': dataset,
                'Alpha': alpha,
                'Theta': cfg.FIXED_THETA,
                'Execution Time': elapsed
            }]).to_csv(time_file, mode='a', header=False, index=False)

            # remove from checkpoint & persist
            params['alphas'].pop(0)
            with open(cp_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        # — Experiment 2: remaining thetas —
        thetas = params.get('thetas', [])
        for theta in tqdm(thetas.copy(),
                          desc=f"{dataset} ⟶ thetas ({len(thetas)} remaining)",
                          leave=False):
            start = time.time()
            ces_naive, ces_optimized  = ce.parallelized_counterfactual_generator(
                ce_dir,
                save_counterfactuals=False,
                processes=6,
                alpha=cfg.FIXED_ALPHA,
                theta=theta
            )
            elapsed = time.time() - start

            # Convert Solution arrays to space-separated strings for CSV compatibility
            ces_naive["Solution"] = ces_naive["Solution"].apply(array_to_string)
            ces_optimized["Solution"] = ces_optimized["Solution"].apply(array_to_string)

            # save CFs + timing
            ces_naive.to_csv(ce_dir / f"confetti_naive_{dataset}_resnet_theta_{theta}.csv", index=False)
            ces_optimized.to_csv(ce_dir / f"confetti_optimized_{dataset}_resnet_theta_{theta}.csv", index=False)
            pd.DataFrame([{
                'Dataset': dataset,
                'Alpha': cfg.FIXED_ALPHA,
                'Theta': theta,
                'Execution Time': elapsed
            }]).to_csv(time_file, mode='a', header=False, index=False)

            # remove from checkpoint & persist
            params['thetas'].pop(0)
            with open(cp_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
