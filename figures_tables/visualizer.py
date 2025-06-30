import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def boxplot_tradeoff(data: pd.DataFrame, metric1: Optional[str] = 'sparsity',
                     metric2: Optional[str] = 'confidence', parameter : Optional[str] = 'alpha') -> None:
    """
    Create a boxplot to visualize trade-offs involving alpha and theta parameters.
    This function assumes that data has already been preprocessed to being grouped by the corresponding parameter.
    Supported trade-offs:
    - Sparsity vs Confidence
    - Confidence vs Coverage
    - Sparsity vs Proximity
    - Confidence vs Proximity

    Parameters:
    - data (pd.DataFrame): DataFrame containing summary results from experiments.
    - metric1 (str): One of ['sparsity', 'confidence'].
    - metric2 (str): One of ['confidence', 'coverage', 'proximity'].
    - parameter Optional[str]: Only applicable for Sparsity vs Confidence. Which parameter to use for illustration.
    Defaults to alpha, but it can also be "theta".
    """
    valid_metrics1 = ['sparsity', 'confidence']
    valid_metrics2 = ['confidence', 'coverage', 'proximity']
    if metric1 not in valid_metrics1 or metric2 not in valid_metrics2:
        raise ValueError("Invalid metrics. metric1 must be in ['sparsity', 'confidence'], "
                         "metric2 in ['confidence', 'coverage', 'proximity'].")

    y_metric, x_param = None, None

    if metric1 == 'sparsity' and metric2 == 'confidence' and parameter == 'alpha':
        y_metric = 'Sparsity'
        x_param = 'Param Config'
        title = "Trade-off: Sparsity vs Confidence (α control)"
        xlabel = r"$\alpha$ (Confidence weight)"
        ylabel = "Sparsity"

    elif metric1 == 'sparsity' and metric2 == 'confidence' and parameter == 'theta':
        y_metric = 'Sparsity'
        x_param = 'Param Config'
        title = "Trade-off: Sparsity vs Confidence (θ control)"
        xlabel = r"$\theta$ (Confidence threshold)"
        ylabel = "Sparsity"

    elif metric1 == 'confidence' and metric2 == 'coverage':
        y_metric = 'Coverage'
        x_param = 'Param Config'
        title = "Trade-off: Confidence vs Coverage (θ control)"
        xlabel = r"$\theta$ (Confidence threshold)"
        ylabel = "Coverage"

    elif metric1 == 'sparsity' and metric2 == 'proximity':
        y_metric = 'Proximity DTW'
        x_param = 'Param Config'
        title = "Trade-off: Sparsity vs Proximity (α control)"
        xlabel = r"$\alpha$ (Confidence weight)"
        ylabel = "Proximity (DTW)"

    elif metric1 == 'confidence' and metric2 == 'proximity':
        y_metric = 'Proximity DTW'
        x_param = 'Param Config'
        title = "Trade-off: Confidence vs Proximity (θ control)"
        xlabel = r"$\theta$ (Confidence threshold)"
        ylabel = "Proximity (DTW)"

    else:
        raise ValueError("Unsupported metric combination.")

    # Apply consistent plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot with consistent color
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_param, y=y_metric, color='#64cdc0', linewidth=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def boxplot_all_tradeoffs(results_alpha: pd.DataFrame,
                          results_theta: pd.DataFrame) -> None:
    """
    Draws a 3×2 grid of boxplots for six trade-offs, sourcing data
    from the appropriate results DataFrame.

    Parameters
    ----------
    results_alpha : pd.DataFrame
        Experiment summaries where α (confidence weight) is varied.
    results_theta : pd.DataFrame
        Experiment summaries where θ (confidence threshold) is varied.
    """
    tradeoffs = [
        ('Sparsity', 'Param Config',
         'i. Sparsity vs Confidence (α control)',
         r'$\alpha$ (Confidence weight)', 'Sparsity', results_alpha),

        ('Sparsity', 'Param Config',
         'ii. Sparsity vs Confidence (θ control)',
         r'$\theta$ (Confidence threshold)', 'Sparsity', results_theta),

        ('Proximity DTW', 'Param Config',
         'iii. Sparsity vs Proximity DTW (α control)',
         r'$\alpha$ (Confidence weight)', 'Proximity (DTW)', results_alpha),

        ('Proximity DTW', 'Param Config',
         'iv. Confidence vs Proximity (θ control)',
         r'$\theta$ (Confidence threshold)', 'Proximity (DTW)', results_theta),

        ('Coverage', 'Param Config',
         'v. Sparsity vs Coverage (α control)',
         r'$\alpha$ (Confidence weight)', 'Coverage', results_alpha),

        ('Coverage', 'Param Config',
         'vi. Confidence vs Coverage (θ control)',
         r'$\theta$ (Confidence threshold)', 'Coverage', results_theta),
    ]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=False)
    axes = axes.flatten()

    for ax, (y_metric, x_param, title, xlabel, ylabel, df) in zip(axes, tradeoffs):
        sns.boxplot(
            data=df,
            x=x_param,
            y=y_metric,
            ax=ax,
            linewidth=0.6,
            boxprops={'facecolor': '#64cdc0'}
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.9)

    plt.tight_layout()
    plt.show()




def plot_with_cam(sample: np.ndarray,
                  counterfactual: np.ndarray,
                  weights: np.ndarray = None,
                  dimension: int = 0) -> pd.DataFrame:
    """
    Visualizes a selected dimension of the counterfactual time series against the original series,
    optionally including Class Activation Map (CAM) weights.

    Parameters:
    - sample (np.ndarray): Original input time series with shape (timesteps, features).
    - counterfactual (np.ndarray): Counterfactual time series with shape (timesteps, features).
    - weights (np.ndarray or None): Class Activation Map (CAM) weights (default: None).
    - dimension (int): Index of the dimension to visualize (default: 0).

    Returns:
    - pd.DataFrame: A DataFrame containing original, counterfactual, and optional CAM weights for the selected dimension.
    """

    # Reshape for time-series format: [time, dimension] → [dim, time]
    sample_reshaped, counterfactual_reshaped = sample.T, counterfactual.T

    if weights is not None:
        weights = np.interp(weights, (weights.min(), weights.max()), (-10, 10))

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(15, 3), sharex=True)

    original_color, cf_color, cam_color = '#64cdc0', '#ff595a', '#ffa500'

    # Plot original series
    ax.plot(sample_reshaped[dimension], label='Original Time Series', color=original_color, linewidth=2)

    # Plot counterfactual differences
    diffs = np.where(sample_reshaped[dimension] != counterfactual_reshaped[dimension])[0]
    if diffs.size > 0:
        start_diff, end_diff = diffs[0], diffs[-1]
        ax.plot(range(start_diff, end_diff + 1), counterfactual_reshaped[dimension, start_diff:end_diff + 1],
                label='Counterfactual Segment', color=cf_color, linewidth=2, linestyle='--')

        if start_diff > 0:
            ax.plot([start_diff - 1, start_diff],
                    [sample_reshaped[dimension, start_diff - 1], counterfactual_reshaped[dimension, start_diff]],
                    color=cf_color, linewidth=2)
        if end_diff < sample_reshaped[dimension].size - 1:
            ax.plot([end_diff, end_diff + 1],
                    [counterfactual_reshaped[dimension, end_diff], sample_reshaped[dimension, end_diff + 1]],
                    color=cf_color, linewidth=2)

    if weights is not None:
        ax.plot(weights, label='CAM Weights', color=cam_color, linewidth=2, linestyle=':')

    ax.set_title(f"Dimension {dimension + 1}", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    legend_handles = [ax.lines[0]]
    if diffs.size > 0:
        legend_handles.append(ax.lines[1])
    if weights is not None:
        legend_handles.append(ax.lines[-1])
    ax.legend(handles=legend_handles, fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Output DataFrame
    return pd.DataFrame({
        f'Original_Dim_{dimension + 1}': sample_reshaped[dimension],
        f'Counterfactual_Dim_{dimension + 1}': counterfactual_reshaped[dimension],
        **({'CAM_Weights': weights} if weights is not None else {})
    })

def visualize_counterfactuals(sample: np.ndarray,
                               counterfactual: np.ndarray,
                               dimension: int = 0) -> None:
    """
    Visualizes the counterfactual time series against the original series for a given instance
    in a selected dimension.

    Parameters:
    - sample (np.ndarray): Original input time series with shape (timesteps, features).
    - counterfactual (np.ndarray): Generated counterfactual time series with same shape as sample.
    - dimension (int): Index of the dimension to visualize (default: 0).

    Returns:
    - None
    """
    # Reshape to [dimension, time]
    sample_reshaped = sample.T
    counterfactual_reshaped = counterfactual.T

    # Prepare plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 3))

    # Plot original series
    original_line, = ax.plot(sample_reshaped[dimension], label='Original', color='#64cdc0', linewidth=2)

    # Identify differences
    diffs = np.where(sample_reshaped[dimension] != counterfactual_reshaped[dimension])[0]

    if diffs.size > 0:
        start_diff, end_diff = diffs[0], diffs[-1]

        counterfactual_line, = ax.plot(
            range(start_diff, end_diff + 1),
            counterfactual_reshaped[dimension, start_diff:end_diff + 1],
            label='Counterfactual',
            color='#ff595a',
            linewidth=2,
            linestyle='--'
        )

        if start_diff > 0:
            ax.plot([start_diff - 1, start_diff],
                    [sample_reshaped[dimension, start_diff - 1], counterfactual_reshaped[dimension, start_diff]],
                    color='salmon', linewidth=2)

        if end_diff < sample_reshaped[dimension].size - 1:
            ax.plot([end_diff, end_diff + 1],
                    [counterfactual_reshaped[dimension, end_diff], sample_reshaped[dimension, end_diff + 1]],
                    color='salmon', linewidth=2)

        ax.legend(handles=[original_line, counterfactual_line], fontsize=12)
    else:
        ax.legend(handles=[original_line], fontsize=12)

    ax.set_title(f"Dimension {dimension + 1}", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def compare_counterfactuals(sample: np.ndarray,
                             counterfactuals: list[np.ndarray],
                             dimension: int = 0,
                             titles: list[str] = None) -> None:
    """
    Compares three counterfactuals for the same sample in a selected dimension.

    Parameters:
    - sample (np.ndarray): Original input time series with shape (timesteps, features).
    - counterfactuals (list of np.ndarray): List of 3 counterfactuals to compare.
    - dimension (int): Index of the dimension to visualize (default: 0).
    - titles (list of str, optional): Custom titles for each subplot (must be length 3).

    Returns:
    - None
    """
    assert len(counterfactuals) == 3, "Exactly 3 counterfactuals must be provided."
    if titles is not None:
        assert len(titles) == 3, "titles must have exactly 3 elements if provided."
    else:
        titles = [f"Comparison {i + 1}" for i in range(3)]

    sample_reshaped = sample.T
    counterfactuals_reshaped = [cf.T for cf in counterfactuals]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    for idx, (cf_reshaped, ax) in enumerate(zip(counterfactuals_reshaped, axes)):
        original_line, = ax.plot(sample_reshaped[dimension], label='Original', color='#64cdc0', linewidth=2)
        diffs = np.where(sample_reshaped[dimension] != cf_reshaped[dimension])[0]

        if diffs.size > 0:
            start_diff, end_diff = diffs[0], diffs[-1]
            counterfactual_line, = ax.plot(
                range(start_diff, end_diff + 1),
                cf_reshaped[dimension, start_diff:end_diff + 1],
                label='Counterfactual',
                color='#ff595a',
                linewidth=2,
                linestyle='--'
            )

            if start_diff > 0:
                ax.plot([start_diff - 1, start_diff],
                        [sample_reshaped[dimension, start_diff - 1], cf_reshaped[dimension, start_diff]],
                        color='salmon', linewidth=2)

            if end_diff < sample_reshaped[dimension].size - 1:
                ax.plot([end_diff, end_diff + 1],
                        [cf_reshaped[dimension, end_diff], sample_reshaped[dimension, end_diff + 1]],
                        color='salmon', linewidth=2)

            ax.legend(handles=[original_line, counterfactual_line], fontsize=12)
        else:
            ax.legend(handles=[original_line], fontsize=12)

        ax.set_title(f"{titles[idx]} - Dimension {dimension + 1}", fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True)

    plt.tight_layout()
    plt.show()



def main():
    df = pd.read_csv('/Users/alan.paredes/Desktop/confetti/benchmark/evaluations/all_evaluation_results.csv')
    column_order = ['Explainer', 'Model', 'Dataset', 'Alpha', 'Param Config', 'Coverage', 'Validity', 'Confidence',
                    'Sparsity', 'Proximity L1', 'Proximity L2', 'Proximity DTW', 'yNN']
    results = df[column_order]
    # Obtain all the rows where the Explainer contains somehow 'confetti'
    confetti_results = results[results['Explainer'].str.contains('confetti', case=False, na=False)]
    results_alphas = confetti_results[confetti_results['Alpha'] == True]
    results_thetas = confetti_results[confetti_results['Alpha'] == False]
    # Create boxplots for the trade-offs
    boxplot_all_tradeoffs(results_alphas, results_thetas)


if __name__ == "__main__":
    main()