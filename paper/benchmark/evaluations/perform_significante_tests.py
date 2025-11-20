from rankings import normalize_proximity_metrics, do_statistical_test
from paper import config as cfg
import pandas as pd

def benchmark_significance_tests():
    models = ["fcn", "resnet"]
    results_dir = cfg.EVALUATIONS_FILE
    benchmark_summary : pd.DataFrame = pd.read_csv(results_dir)

    competitors = ["Comte", "TSEvo", "Sets"]
    confetti_method_map = {
        "Confetti (alpha=0.5)": "Confetti α=0.5",
        "Confetti (theta=0.95)": "Confetti θ=0.95",
        "Confetti (alpha=0.0)": "Confetti α=0.0",
    }
    allowed = competitors + list(confetti_method_map.keys())
    cleaned = benchmark_summary[benchmark_summary["Explainer"].isin(allowed)].copy()
    cleaned["Explainer"] = cleaned["Explainer"].replace(confetti_method_map)

    cleaned = cleaned.drop(columns=["Param Config", "Alpha"])
    for model in models:
        filtered = cleaned[cleaned["Model"] == model].copy()
        filtered = normalize_proximity_metrics(filtered)
        do_statistical_test(
            filtered,
            alpha=0.05,
            output_excel=f"benchmark_{model}_rankings_new.xlsx",
        )


def execution_times_significance_test():
    models = ["fcn", "resnet", "logistic"]
    competitors = ["comte", "tsevo", "sets"]
    confetti_method_map = {
        "Confetti α=0.5":{"alpha":0.5, "theta":0.51},
        "Confetti θ=0.95":{"alpha":0.5, "theta":0.95},
        "Confetti α=0.0": {"alpha":0.0, "theta":0.51},
    }
    allowed = competitors + list(confetti_method_map.keys())
    for model in models:
        execution_times = pd.DataFrame()
        for method in allowed:
            if method.startswith("Confetti"):
                exec_times_dir = cfg.RESULTS_DIR / f"execution_time_confetti_{model}.csv"
                times = pd.read_csv(exec_times_dir)
                # Select the corresponding alpha and theta
                actual_method = times[(times["Alpha"] == confetti_method_map[method]["alpha"]) &
                                        (times["Theta"] == confetti_method_map[method]["theta"])]
                cleaned_execution_times = actual_method.drop(columns=["Alpha", "Theta"])
                cleaned_execution_times["Explainer"] = method
                execution_times = pd.concat([execution_times, cleaned_execution_times], axis=0)
            else:
                exec_times_dir = cfg.RESULTS_DIR / f"execution_time_{method}_{model}.csv"
                times = pd.read_csv(exec_times_dir)
                times["Explainer"] = method.capitalize()
                times = times.rename(columns={"Execution Time (seconds)": "Execution Time"})
                execution_times = pd.concat([execution_times, times], axis=0)

        print(execution_times)
        #Check nans
        print(f"Model: {model}, NaNs in execution times: {execution_times['Execution Time'].isna().sum()}")
        do_statistical_test(
            execution_times,
            execution_times=True,
            alpha=0.05,
            output_excel=f"execution_times_{model}_rankings.xlsx",
        )


def main():
    benchmark_significance_tests()
    #execution_times_significance_test()

if __name__ == "__main__":
    main()