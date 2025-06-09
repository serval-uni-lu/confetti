import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluator import Evaluator

DATASETS = [
    'ArticularyWordRecognition', 'BasicMotions', 'Epilepsy',
    'ERing', 'Libras', 'NATOPS', 'RacketSports'
]
CONFETTI_ALPHA = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
CONFETTI_THETA = [0.55, 0.65, 0.75, 0.85, 0.95]
EXPLAINERS = [
    ("comte", "Comte"),
    ("sets", "Sets"),
    ("tsevo", "TSEvo"),
]
MODELS = ['fcn', 'resnet']

def eval_config(args):
    """Parallel evaluation function. Skips if CSV not found."""
    explainer, dataset, model, display_name, extra_kwargs = args
    ev = Evaluator()
    try:
        _, summary = ev.evaluate_from_csv(explainer, dataset, model, **extra_kwargs)
        summary["Explainer"] = display_name
        summary["Model"] = model
        return summary
    except FileNotFoundError:
        print(f"File not found for {display_name} - {dataset} - {model} - {extra_kwargs}. Skipping.")
        return None
    except Exception as e:
        print(f"Failed for {display_name} - {dataset} - {model} - {extra_kwargs}: {e}")
        return None

def evaluate_results_parallel():
    configs = []
    # Baseline explainers (comte, sets, tsevo)
    for explainer, display_name in EXPLAINERS:
        for dataset in DATASETS:
            for model in MODELS:
                configs.append((explainer, dataset, model, display_name, {}))
    # Confetti (naive/optimized) for alpha
    for confetti in ['confetti_naive', 'confetti_optimized']:
        for dataset in DATASETS:
            for model in MODELS:
                for alpha in CONFETTI_ALPHA:
                    extra_kwargs = {"alpha": True, "param_config": alpha}
                    display_name = f"{confetti.replace('_', ' ').title()} (alpha={alpha})"
                    configs.append((confetti, dataset, model, display_name, extra_kwargs))
    # Confetti (naive/optimized) for theta
    for confetti in ['confetti_naive', 'confetti_optimized']:
        for dataset in DATASETS:
            for model in MODELS:
                for theta in CONFETTI_THETA:
                    extra_kwargs = {"alpha": False, "param_config": theta}
                    display_name = f"{confetti.replace('_', ' ').title()} (theta={theta})"
                    configs.append((confetti, dataset, model, display_name, extra_kwargs))

    summaries = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(eval_config, cfg) for cfg in configs]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                summaries.append(result)

    if summaries:
        final_df = pd.concat(summaries, ignore_index=True, sort=True)  # <-- This ensures all columns are kept
        print(final_df)
        # You can save the results if you want:
        final_df.to_csv("all_evaluation_results.csv", index=False)
    else:
        print("No valid results found.")

def main():
    evaluate_results_parallel()

if __name__ == "__main__":
    main()
