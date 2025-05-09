import config as cfg
import keras
from pathlib import Path
import warnings
import confetti.CAM.class_activation_map as cam
from confetti.explainer.confetti_explainer import CONFETTI
from confetti.explainer.utils import load_data, get_samples
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

    for dataset, params in checkpoint.items():
        ce_dir = cfg.RESULTS_DIR / dataset
        ce_dir.mkdir(parents=True, exist_ok=True)

        log_file = ce_dir / "execution_times.csv"
        if not log_file.exists():
            pd.DataFrame(columns=['Dataset','Alpha','Theta','Execution Time']) \
              .to_csv(log_file, index=False)

        # load data, model, explainer
        X_train, X_test, y_train, y_test = load_data(dataset, one_hot=False)
        model = keras.models.load_model(
            str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_fcn.keras")
        )
        if dataset in cfg.NUMBER_OF_SAMPLES_PER_CLASS:
            X_samples, y_samples = get_samples(dataset, one_hot=False)
        else:
            X_samples, y_samples = X_test, y_test

        training_weights = cam.compute_weights_cam(
            model, X_train, dataset=dataset,
            save_weights=True, data_type='training'
        )
        ce = CONFETTI(
            str(cfg.TRAINED_MODELS_DIR / dataset / f"{dataset}_fcn.keras"),
            X_train, X_samples, y_samples, y_train, training_weights
        )

        # — Experiment 1: remaining alphas —
        alphas = params.get('alphas', [])
        for alpha in tqdm(alphas.copy(),
                          desc=f"{dataset} ⟶ alphas ({len(alphas)} remaining)",
                          leave=False):
            start = time.time()
            ces = ce.parallelized_counterfactual_generator(
                ce_dir,
                save_counterfactuals=True,
                processes=4,
                alpha=alpha,
                theta=cfg.FIXED_THETA
            )
            elapsed = time.time() - start

            # save CFs + timing
            pd.DataFrame(ces).to_csv(
                ce_dir / f"ces_{dataset}_alpha_{alpha}.csv", index=False
            )
            pd.DataFrame([{
                'Dataset': dataset,
                'Alpha': alpha,
                'Theta': cfg.FIXED_THETA,
                'Execution Time': elapsed
            }]).to_csv(log_file, mode='a', header=False, index=False)

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
            ces = ce.parallelized_counterfactual_generator(
                ce_dir,
                save_counterfactuals=True,
                processes=4,
                alpha=cfg.FIXED_ALPHA,
                theta=theta
            )
            elapsed = time.time() - start

            # save CFs + timing
            pd.DataFrame(ces).to_csv(
                ce_dir / f"ces_{dataset}_theta_{theta}.csv", index=False
            )
            pd.DataFrame([{
                'Dataset': dataset,
                'Alpha': cfg.FIXED_ALPHA,
                'Theta': theta,
                'Execution Time': elapsed
            }]).to_csv(log_file, mode='a', header=False, index=False)

            # remove from checkpoint & persist
            params['thetas'].pop(0)
            with open(cp_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
