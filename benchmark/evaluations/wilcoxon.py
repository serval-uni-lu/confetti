import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import config as cfg



def significance_ranking(
        df: pd.DataFrame,
        metric: str,
        higher_better: bool = True,
        alpha: float = 0.05,
        method_kw: str = "exact",           # "exact" or "asymptotic"
        group_col: str = "Dataset",
        explainer_col: str = "Explainer",
) -> pd.DataFrame:
    """
    Give each method a rank, but only split ranks when the paired
    Wilcoxon test between groups is significant at `alpha`.

    Steps
    -----
    1. Compute the mean metric per method and sort best→worst.
    2. Walk down the list, slotting each method into the first
       existing group for which *all* comparisons yield p>alpha.
       If none qualify, start a new group.
    3. Return a table: mean_score | rank
    """
    # Performance ordering
    mean_scores = (
        df.groupby(explainer_col)[metric]
          .mean().round(2)
          .sort_values(ascending=not higher_better)
    )
    ordered = mean_scores.index.tolist()

    # helper: paired p‑value between two methods
    def p_val(m1, m2):
        wide = (df.loc[df[explainer_col].isin([m1, m2]),
                       [group_col, explainer_col, metric]]
                  .pivot(index=group_col, columns=explainer_col, values=metric)
                  .dropna()).round(2)
        if wide.empty:
            return np.nan
        return wilcoxon(wide[m1], wide[m2],
                        alternative="two-sided",
                        method=method_kw).pvalue

    # Build significance groups
    groups:list[set[str]] = []
    for m in ordered:
        placed = False
        for g in groups:
            if all(p_val(m, gm) > alpha for gm in g):
                g.add(m)                       # same rank
                placed = True
                break
        if not placed:
            groups.append({m})                 # new (worse) rank

    # Assign ranks
    rank_map = {m: r+1 for r, g in enumerate(groups) for m in g}
    out = pd.DataFrame({
        explainer_col: mean_scores.index,
        f"mean_{metric}": mean_scores.values,
        "rank": [rank_map[m] for m in mean_scores.index]
    })
    return out.sort_values("rank")



def strict_significance_ranking(df: pd.DataFrame,
                                alpha: float = 0.05,
                                method_kw: str = "exact") -> pd.DataFrame:
    """
    Rank methods based on strict dominance:
    Method A is ranked higher than Method B only if:
      1) For every dataset, A's score > B's score
      2) The Wilcoxon test across datasets between A and B is significant (p <= alpha)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one column 'Dataset' and one column per method.
    alpha : float
        Significance threshold.
    method_kw : str
        'exact' or 'asymptotic' for scipy.stats.wilcoxon.

    Returns
    -------
    pd.DataFrame
        Columns: Method | rank
    """
    methods = [c for c in df.columns if c != "Dataset" and "Unnamed" not in c]

    def strictly_better(m1, m2):
        x = df[m1]
        y = df[m2]
        # Remove datasets where either value is NaN
        mask = ~(x.isna() | y.isna())
        x = x[mask].values
        y = y[mask].values
        # If no data left, cannot compare
        if len(x) == 0:
            return False
        # Check strict dominance on the remaining datasets
        if not np.all(x > y):
            return False
        # Wilcoxon test
        stat = wilcoxon(x, y, alternative="greater", method=method_kw)
        return stat.pvalue <= alpha

    # Build dominance relationships
    dominance = {m: set() for m in methods}
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                continue
            if strictly_better(m1, m2):
                dominance[m1].add(m2)

    # Compute ranks using layers of dominance
    remaining = set(methods)
    rank = 1
    ranks = {}
    while remaining:
        current_rank = []
        for m in remaining:
            if not any((m in dominance[o]) for o in remaining if o != m):
                current_rank.append(m)
        for m in current_rank:
            ranks[m] = rank
        remaining -= set(current_rank)
        rank += 1

    return pd.DataFrame({"Method": list(ranks.keys()),
                         "rank": list(ranks.values())}).sort_values("rank")


def overall_scores():
    df = pd.read_csv(cfg.EVALUATIONS_FILE)
    # ---- setup ---------------------------------------------------------------
    confetti_method_map = {
        'Confetti Optimized (alpha=0.5)': 'Confetti α=0.5',
        'Confetti Optimized (theta=0.95)': 'Confetti θ=0.95',
        'Confetti Optimized (alpha=0.0)': 'Confetti α=0.0',
    }

    core_explainers = ['Comte', 'TSEvo', 'Sets']

    # ---- 1. keep only the rows you want -------------------------------------
    allowed = core_explainers + list(confetti_method_map.keys())
    cleaned = df[df['Explainer'].isin(allowed)].copy()

    # ---- 2. rename the Confetti variants ------------------------------------
    cleaned['Explainer'] = cleaned['Explainer'].replace(confetti_method_map)

    # ---- 3. drop the now‑irrelevant columns ---------------------------------
    cleaned = cleaned.drop(columns=['Param Config', 'Alpha'])


    fcn_results = cleaned[cleaned['Model'] == 'fcn']
    resnet_results = cleaned[cleaned['Model'] == 'resnet']

    metric_hb_map = {
        "Coverage": True,
        "Validity": True,
        "Sparsity": True,
        "Confidence": True,
        "yNN": True,
        "Proximity L1": False,
        "Proximity L2": False,
        "Proximity DTW": False,
    }

    for metric, hb in metric_hb_map.items():
        print(f"\n=== {metric.upper()} ===")

        # ResNet results
        table_resnet = significance_ranking(
            resnet_results,
            metric,
            higher_better=hb,
            alpha=0.05,
            method_kw="exact",
        )

        # FCN results
        table_fcn = significance_ranking(
            fcn_results,
            metric,
            higher_better=hb,
            alpha=0.05,
            method_kw="exact",
        )

        print(f"{metric} Table ResNet:")
        print(table_resnet)

        print(f"{metric} Table FCN:")
        print(table_fcn)

def sparsity_ranks():
    df = pd.read_csv(cfg.EVALUATIONS_FILE)
    # ---- setup ---------------------------------------------------------------
    confetti_method_map = {
        'Confetti Optimized (alpha=0.5)': 'Confetti α=0.5',
        'Confetti Optimized (theta=0.95)': 'Confetti θ=0.95',
        'Confetti Optimized (alpha=0.0)': 'Confetti α=0.0',
    }

    core_explainers = ['Comte', 'TSEvo', 'Sets']

    # ---- 1. keep only the rows you want -------------------------------------
    allowed = core_explainers + list(confetti_method_map.keys())
    cleaned = df[df['Explainer'].isin(allowed)].copy()

    # ---- 2. rename the Confetti variants ------------------------------------
    cleaned['Explainer'] = cleaned['Explainer'].replace(confetti_method_map)

    # ---- 3. drop the now‑irrelevant columns ---------------------------------
    cleaned = cleaned.drop(columns=['Param Config', 'Alpha'])
    # Keep only relevant columns and drop rows without Sparsity
    sparsity_df = cleaned[['Dataset', 'Explainer', 'Sparsity']].dropna(subset=['Sparsity'])

    pivot_table_sparsity = sparsity_df[sparsity_df['Explainer'] != 'SETS']
    # Pivot to wide format without removing duplicates
    pivot_table_sparsity = pivot_table_sparsity.pivot_table(index='Dataset', columns='Explainer', values='Sparsity',
                                                            aggfunc='mean').reset_index()

    # Reorder columns: Dataset + non-Confetti + Confetti α=...
    columns = pivot_table_sparsity.columns.tolist()
    confetti_cols = [col for col in columns if
                     isinstance(col, str) and (col.startswith("Confetti α") or col.startswith("Confetti θ"))]
    non_confetti_cols = [col for col in columns if col != 'Dataset' and col not in confetti_cols]
    ordered_cols = ['Dataset'] + sorted(non_confetti_cols) + sorted(confetti_cols)
    pivot_table_sparsity = pivot_table_sparsity[ordered_cols]
    pivot_table_sparsity.columns.name = None

    # Format floats to 3 decimal places
    sparsity_table = pivot_table_sparsity.round(2)
    sparsity_scores = strict_significance_ranking(sparsity_table)
    print("\n=== Sparsity Scores ===")
    print(sparsity_scores)

def confidence_ranks():
    df = pd.read_csv(cfg.EVALUATIONS_FILE)
    # ---- setup ---------------------------------------------------------------
    confetti_method_map = {
        'Confetti Optimized (alpha=0.5)': 'Confetti α=0.5',
        'Confetti Optimized (theta=0.95)': 'Confetti θ=0.95',
        'Confetti Optimized (alpha=0.0)': 'Confetti α=0.0',
    }

    core_explainers = ['Comte', 'TSEvo', 'Sets']

    # ---- 1. keep only the rows you want -------------------------------------
    allowed = core_explainers + list(confetti_method_map.keys())
    cleaned = df[df['Explainer'].isin(allowed)].copy()

    # ---- 2. rename the Confetti variants ------------------------------------
    cleaned['Explainer'] = cleaned['Explainer'].replace(confetti_method_map)

    # ---- 3. drop the now‑irrelevant columns ---------------------------------
    cleaned = cleaned.drop(columns=['Param Config', 'Alpha'])

    # Keep only relevant columns
    confidence_df = cleaned[['Dataset', 'Explainer', 'Confidence']]

    pivot_table_confidence = confidence_df[confidence_df['Explainer'] != 'TSEvo']

    # Pivot to wide format without removing duplicates
    pivot_table_confidence = pivot_table_confidence.pivot_table(index='Dataset', columns='Explainer',
                                                                values='Confidence', aggfunc='mean').reset_index()

    # Reorder columns: Dataset + non-Confetti + Confetti α=...
    columns = pivot_table_confidence.columns.tolist()

    confetti_cols = [col for col in columns if
                     isinstance(col, str) and (col.startswith("Confetti α") or col.startswith("Confetti θ"))]
    non_confetti_cols = [col for col in columns if col != 'Dataset' and col not in confetti_cols]
    ordered_cols = ['Dataset'] + sorted(non_confetti_cols) + sorted(confetti_cols)

    pivot_table_confidence = pivot_table_confidence[ordered_cols]
    pivot_table_confidence.columns.name = None

    # Format floats to 3 decimal places
    confidence_table = pivot_table_confidence.round(2)
    confidence_scores = strict_significance_ranking(confidence_table)
    print("\n=== Confidence Scores ===")
    print(confidence_scores)


def main():
    #overall_scores()
    sparsity_ranks()
    confidence_ranks()

if __name__ == '__main__':
    main()
