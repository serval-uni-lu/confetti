import numpy as np

from paper.benchmark.evaluations.rankings import normalize_proximity_metrics, wilcoxon_holm, do_statistical_test
from paper import config as cfg
import pandas as pd

def experiment_objectives():
    results_dir = cfg.EXPERIMENT_OBJECTIVES / "objectives_summary.csv"
    objectives_summary : pd.DataFrame = pd.read_csv(results_dir)

    results = normalize_proximity_metrics(objectives_summary)
    do_statistical_test(
        results,
        alpha=0.05,
        output_excel=f"rankings_objectives.xlsx",
    )

def experiment_proximity_metric():
    results_dir = cfg.EXPERIMENT_PROXIMITY_METRIC / "best_proximity_metric.csv"
    proximity_metric_summary : pd.DataFrame = pd.read_csv(results_dir)
    #results = normalize_proximity_metrics(proximity_metric_summary)
    print(proximity_metric_summary.columns)
    do_statistical_test(
        proximity_metric_summary,
        alpha=0.05,
        output_excel=f"rankings_proximity_metric.xlsx",
    )

def experiment_execution_times_per_shape():
    models = ["fcn", "resnet"]
    for model in models:
        results_dir = cfg.EXPERIMENT_EXECUTION_TIMES_PER_SHAPE / f"execution_times_per_shape_{model}"
        execution_times : pd.DataFrame = pd.read_csv(results_dir)
        results = normalize_proximity_metrics(execution_times)
        do_statistical_test(
            results,
            alpha=0.05,
            output_excel=f"rankings_execution_times_per_shape_{model}.xlsx",
        )


def experiment_ablation_study():
    output_path = cfg.EXPERIMENT_ABLATION_STUDY / "ablation_study_significance_tests.xlsx"
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

    for model in ["resnet", "fcn"]:
        results_dir = cfg.EXPERIMENT_ABLATION_STUDY / f"ablation_summary_{model}.csv"
        results = pd.read_csv(results_dir)

        # Identify metric columns
        metric_columns = [col for col in results.columns if col not in ["Explainer", "Dataset"]]

        rows = []
        for metric in metric_columns:
            ablation = results[results["Explainer"].str.contains("Ablation")][metric].values
            normal = results[results["Explainer"].str.contains("Normal")][metric].values

            test_results = wilcoxon_holm(ablation, normal)

            row = {
                "Metric": metric,
                "p_value": test_results["p_value_holm"],
                "Ablation": np.mean(ablation),
                "Normal": np.mean(normal),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=model, index=False)

    writer.close()
    print(f"Saved results to {output_path}")


def main():
    #experiment_objectives()
    experiment_proximity_metric()
    #experiment_execution_times_per_shape()
    #experiment_ablation_study()

if __name__ == "__main__":
    main()


