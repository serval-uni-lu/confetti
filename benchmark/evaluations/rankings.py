from typing import Tuple

import pandas as pd
import numpy as np
from pingouin import friedman
import scikit_posthocs as sp
import config as cfg
from sklearn.preprocessing import MinMaxScaler

def normalize_proximity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize proximity metrics (L1, L2, DTW) to [0,1] within each Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'Dataset' and proximity metric columns:
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
        data[col + " Norm"] = (
            data.groupby("Dataset")[col]
              .transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten())
        )
    return data

def rank_data(df: pd.DataFrame,
              metric: str,
              higher_is_better: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank methods per dataset for a given metric.
    """
    pivot = df.pivot(index="Dataset", columns="Explainer", values=metric).dropna()

    if higher_is_better:
        ranked = pivot.apply(
            lambda row: rankdata(-row.values, method="average"),
            axis=1, result_type="expand"
        )
    else:
        ranked = pivot.apply(
            lambda row: rankdata(row.values, method="average"),
            axis=1, result_type="expand"
        )

    ranked.columns = pivot.columns
    return ranked, pivot

def friedman_test(rank_df: pd.DataFrame) -> float:
    """
    Perform Friedman test on ranked data.
    """
    arrays = [rank_df[col].values for col in rank_df.columns]
    _, p_val = friedmanchisquare(*arrays)
    return p_val


def nemenyi_test(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test.
    """
    nemenyi = sp.posthoc_nemenyi_friedman(rank_df.values)
    nemenyi.index = rank_df.columns
    nemenyi.columns = rank_df.columns
    return nemenyi


def do_statistical_test(results: pd.DataFrame,
                        alpha: float = 0.05,
                        output_excel: str = "nemenyi_resnet.xlsx") -> None:
    """
    Full pipeline: rank -> Friedman -> Nemenyi (ResNet only).
    Save all Nemenyi matrices + average metric values to Excel.
    """
    # filter for ResNet
    resnet_df = results[results["Model"].str.lower() == "resnet"]

    # metrics and whether higher is better
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
        rank_df, pivot = rank_data(resnet_df, metric, higher_is_better=higher_better)
        p_val = friedman_test(rank_df)

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

        # write two sheets: one for p-values, one for descriptive stats
        matrix.to_excel(writer, sheet_name=f"{metric[:25]}_pvals")
        desc.to_excel(writer, sheet_name=f"{metric[:25]}_avg")

    writer.close()
    print(f"Excel with Nemenyi matrices + averages saved to {output_excel}")


def main():
    results = pd.read_csv(cfg.EVALUATIONS_FILE)

    confetti_method_map = {
        'Confetti Optimized (alpha=0.5)': 'Confetti α=0.5',
        'Confetti Optimized (theta=0.95)': 'Confetti θ=0.95',
        'Confetti Optimized (alpha=0.0)': 'Confetti α=0.0',
        'Ablation Study (alpha=0.5)': 'Ablation Study α=0.5',
        'Ablation Study (theta=0.95)': 'Ablation Study θ=0.95',
        'Ablation Study (alpha=0.0)': 'Ablation Study α=0.0',
    }

    filtered_results = results[results['Explainer'].isin(confetti_method_map)].copy()
    filtered_results['Explainer'] = filtered_results['Explainer'].replace(confetti_method_map)

    cleaned_results = filtered_results.drop(columns=['Param Config', 'Alpha'])
    cleaned_results = normalize_proximity_metrics(cleaned_results)

    do_statistical_test(cleaned_results, alpha=0.05, output_excel="nemenyi_resnet.xlsx")

if __name__ == "__main__":
    main()