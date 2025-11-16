from typing import Tuple

import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import rankdata, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from paper import config as cfg
from sklearn.preprocessing import MinMaxScaler


def normalize_proximity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize proximity metrics (L1, L2, DTW) to [0,1] within each Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Object containing 'Dataset' and proximity metric columns:
        - 'Proximity L1'
        - 'Proximity L2'
        - 'Proximity DTW'

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with three new normalized columns:
        - 'Proximity L1_norm'
        - 'Proximity L2_norm'
        - 'Proximity DTW_norm'
    """
    proximity_cols = ["Proximity L1", "Proximity L2", "Proximity DTW"]
    data = df.copy()

    for col in proximity_cols:
        data[col + " Norm"] = data.groupby("Dataset")[col].transform(
            lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
        )
    return data

def rank_data(
    df: pd.DataFrame, metric: str, higher_is_better: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank methods per dataset for a given metric.
    """
    pivot = df.pivot(index="Dataset", columns="Explainer", values=metric).dropna()

    if higher_is_better:
        ranked = pivot.apply(
            lambda row: rankdata(-row.values, method="average"),
            axis=1,
            result_type="expand",
        )
    else:
        ranked = pivot.apply(
            lambda row: rankdata(row.values, method="average"),
            axis=1,
            result_type="expand",
        )

    ranked.columns = pivot.columns
    return ranked, pivot

def friedman_test(rank_df: pd.DataFrame) -> float:
    """
    Perform Friedman test on ranked data.
    Null hypothesis: All methods perform equally on average across datasets.
    """
    arrays = [rank_df[col].values for col in rank_df.columns]
    _, p_val = friedmanchisquare(*arrays)
    return p_val

def nemenyi_test(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test.
    Computes pairwise comparisons between all explainers based on ranks.
    """
    nemenyi = sp.posthoc_nemenyi_friedman(rank_df.values)
    nemenyi.index = rank_df.columns
    nemenyi.columns = rank_df.columns
    return nemenyi

def wilcoxon_holm(method_a: np.ndarray, method_b: np.ndarray) -> dict:
    """
    Perform a Wilcoxon signed-rank test between two methods and apply
    Holm correction to the p-value.

    Parameters
    ----------
    method_a : np.ndarray
        Performance scores of method A across datasets.
    method_b : np.ndarray
        Performance scores of method B across datasets.

    Returns
    -------
    dict
        Dictionary containing the raw Wilcoxon statistic, raw p-value,
        Holm-corrected p-value, and whether the null hypothesis is rejected.
    """
    statistic, p_value = wilcoxon(x=method_a, y=method_b, alternative="two-sided")

    corrected = multipletests(
        pvals=[p_value],
        alpha=0.05,
        method="holm"
    )
    corrected_p = corrected[1][0]
    reject_null = corrected[0][0]

    return {
        "statistic": statistic,
        "p_value_raw": p_value,
        "p_value_holm": corrected_p,
        "reject_null": reject_null,
    }

def do_statistical_test(
    results: pd.DataFrame,
    execution_times: bool = False,
    alpha: float = 0.05,
    output_excel: str = "nemenyi_resnet.xlsx",
) -> None:
    """
    This statisical test is based on the methodology from:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    Journal of Machine Learning Research, 7(Jan), 1-30.
    The approach involves ranking the methods for each dataset,
    performing the Friedman test to check for significant differences (are the average ranks different?),
    and if significant (ranks are significantly different), conducting the Nemenyi post-hoc test (to discover
    which ranks are different). If the Friedman test is not significant, then all methods are considered equivalent.
    Full pipeline: rank -> Friedman -> Nemenyi.
    Saves all Nemenyi matrices + average metric values to Excel.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing evaluation results with columns:
        - 'Model'
        - 'Explainer'
        - Metrics: 'Coverage', 'Validity', 'Sparsity', 'Confidence', 'yNN',
          'Proximity L1 Norm', 'Proximity L2 Norm', 'Proximity DTW Norm'
    model_name : str, optional
        Model to filter for (e.g., "fcn"), by default "resnet"
    alpha : float, optional
        Significance level for the statistical tests, by default 0.05
    save_results : bool, optional
        Whether to save the results to an Excel file, by default True
    output_excel : str, optional
        Path to save the output Excel file, by default "nemenyi_resnet.xlsx"
    """


    # metrics and whether higher is better
    if execution_times:
        metrics = {
            "Execution Time": False,
        }
    else:
        metrics = {
            "Coverage": True,
            "Validity": True,
            "Sparsity": True,
            "Confidence": True,
            "yNN": True,
            "Proximity L1 Norm": False,
            "Proximity L2 Norm": False,
            "Proximity DTW Norm": False,
        }


    writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")

    for metric, higher_better in metrics.items():
        # ranked data + raw values (pivoted)
        rank_df, pivot = rank_data(results, metric, higher_is_better=higher_better)
        p_val : float = friedman_test(rank_df)
        print(f"Ranks for metric {metric} (higher is better: {higher_better}):")
        print(rank_df)

        # descriptive stats
        avg = pivot.mean().to_frame("mean")
        std = pivot.std().to_frame("std")
        desc = avg.join(std)
        desc.index.name = "Explainer"

        if np.isnan(p_val) or p_val >= alpha:
            # not significant → note instead of Nemenyi
            matrix = pd.DataFrame(
                {"Note": [f"Friedman p={p_val:.4f}, not significant"]}
            )
        else:
            matrix = nemenyi_test(rank_df)

        # write three sheets: one for ranks, one for p-values, one for descriptive stats
        rank_df.to_excel(writer, sheet_name=f"{metric[:25]}_ranks")
        matrix.to_excel(writer, sheet_name=f"{metric[:25]}_pvals")
        desc.to_excel(writer, sheet_name=f"{metric[:25]}_avg")


    writer.close()
    print(f"Excel with Nemenyi matrices + averages saved to {output_excel}")




def main():
    results = pd.read_csv(cfg.EVALUATIONS_FILE)

    confetti_method_map = {
        "Confetti Optimized (alpha=0.5)": "Confetti α=0.5",
        "Confetti Optimized (theta=0.95)": "Confetti θ=0.95",
        "Confetti Optimized (alpha=0.0)": "Confetti α=0.0",
        "Ablation Study (alpha=0.5)": "Ablation Study α=0.5",
        "Ablation Study (theta=0.95)": "Ablation Study θ=0.95",
        "Ablation Study (alpha=0.0)": "Ablation Study α=0.0",
    }

    filtered_results = results[results["Explainer"].isin(confetti_method_map)].copy()
    filtered_results["Explainer"] = filtered_results["Explainer"].replace(
        confetti_method_map
    )

    cleaned_results = filtered_results.drop(columns=["Param Config", "Alpha"])
    cleaned_results = normalize_proximity_metrics(cleaned_results)
    print(cleaned_results)

    do_statistical_test(cleaned_results, alpha=0.05, output_excel="nemenyi_resnet.xlsx")


if __name__ == "__main__":
    main()
