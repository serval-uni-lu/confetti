import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def boxplot_single_tradeoff(data: pd.DataFrame, metric1: Optional[str] = 'sparsity',
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
        if ylabel == 'Sparsity':
            ax.set_ylim(0, 1)
        elif ylabel == 'Coverage':
            ax.set_ylim(0, 110)
        else:
            ax.set_ylim(0, 60)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.9)
    # Set min y-axis value to 0 for all plots

    plt.tight_layout()
    plt.show()

#Figure Trade-offs
def boxplot_all_tradeoffs_by_model(results_alpha: pd.DataFrame, results_theta: pd.DataFrame) -> None:
    tradeoffs = [
        ('Sparsity', 'Param Config',
         'i. Sparsity (α control)',
         r'$\alpha$ (Confidence weight)', 'Sparsity', results_alpha),

        ('Sparsity', 'Param Config',
         'ii. Sparsity (θ control)',
         r'$\theta$ (Confidence threshold)', 'Sparsity', results_theta),

        ('Confidence', 'Param Config',
         'iii. Confidence (α control)',
         r'$\alpha$ (Confidence weight)', 'Confidence', results_alpha),

        ('Confidence', 'Param Config',
         'iv. Confidence (θ control)',
         r'$\theta$ (Confidence threshold)', 'Confidence', results_theta),

        ('Proximity DTW', 'Param Config',
         'v. Proximity DTW (α control)',
         r'$\alpha$ (Confidence weight)', 'Proximity (DTW)', results_alpha),

        ('Proximity DTW', 'Param Config',
         'vi. Proximity DTW (θ control)',
         r'$\theta$ (Confidence threshold)', 'Proximity (DTW)', results_theta),
    ]

    palette = {"fcn": "#64CDC0", "resnet": "#ff595a"}

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=False)
    axes = axes.flatten()

    for ax, (y_metric, x_param, title, xlabel, ylabel, df) in zip(axes, tradeoffs):
        sns.boxplot(
            data=df,
            x=x_param,
            y=y_metric,
            hue="Model",
            palette=palette,
            ax=ax,
            linewidth=0.6
        )
        if ylabel == 'Sparsity' or ylabel == 'Confidence':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 60)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.9)
        ax.legend(title='Model', loc='best', fontsize=12, title_fontsize=12)

    plt.tight_layout()

    filename = "tradeoffs_by_model.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

def plot_counterfactual_with_cam(sample: np.ndarray,
                          counterfactual: np.ndarray,
                          weights: np.ndarray = None,
                          dimension: int = 0) -> pd.DataFrame:
    """
    Visualizes a selected dimension of the counterfactual time series against the original series,
    and shows CAM weights as a heatmap below the time series.

    Parameters:
    - sample (np.ndarray): Original input time series with shape (timesteps, features).
    - counterfactual (np.ndarray): Counterfactual time series with shape (timesteps, features).
    - weights (np.ndarray or None): Class Activation Map (CAM) weights (default: None).
    - dimension (int): Index of the dimension to visualize (default: 0).

    Returns:
    - pd.DataFrame: A DataFrame containing original, counterfactual, and optional CAM weights for the selected dimension.
    """
    # Reshape to [dim, time]
    sample_reshaped, counterfactual_reshaped = sample.T, counterfactual.T

    if weights is not None:
        weights = np.interp(weights, (weights.min(), weights.max()), (0, 1))  # Normalize for heatmap

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4.5),
                                   sharex=True,
                                   gridspec_kw={'height_ratios': [3, 0.5]})

    original_color, cf_color = '#64cdc0', '#ff595a'

    # --- Plot 1: Time series ---
    ax1.plot(sample_reshaped[dimension], label='Original Time Series', color=original_color, linewidth=2)

    # Plot counterfactual differences
    diffs = np.where(sample_reshaped[dimension] != counterfactual_reshaped[dimension])[0]
    if diffs.size > 0:
        start_diff, end_diff = diffs[0], diffs[-1]
        ax1.plot(range(start_diff, end_diff + 1), counterfactual_reshaped[dimension, start_diff:end_diff + 1],
                 label='Counterfactual Segment', color=cf_color, linewidth=2, linestyle='--')

        if start_diff > 0:
            ax1.plot([start_diff - 1, start_diff],
                     [sample_reshaped[dimension, start_diff - 1], counterfactual_reshaped[dimension, start_diff]],
                     color=cf_color, linewidth=2)
        if end_diff < sample_reshaped[dimension].size - 1:
            ax1.plot([end_diff, end_diff + 1],
                     [counterfactual_reshaped[dimension, end_diff], sample_reshaped[dimension, end_diff + 1]],
                     color=cf_color, linewidth=2)

    ax1.set_title(f"Dimension {dimension + 1}", fontsize=14)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True)

    # --- Plot 2: Heatmap of CAM weights ---
    if weights is not None:
        ax2.imshow(weights[np.newaxis, :], aspect="auto", cmap="YlOrRd", extent=[0, len(weights), 0, 1])
        ax2.set_yticks([])
        ax2.set_ylabel("CAM", fontsize=10)

    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        f'Original_Dim_{dimension + 1}': sample_reshaped[dimension],
        f'Counterfactual_Dim_{dimension + 1}': counterfactual_reshaped[dimension],
        **({'CAM_Weights': weights} if weights is not None else {})
    })

def plot_counterfactual_with_cam_as_line(sample: np.ndarray,
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

#Figure Method Comparison
def plot_method_comparison_with_cam(sample, counterfactuals_dict, cam, dimension_idx, title=""):
    """
    Plot the original time series and counterfactual segments that differ from the original, for a given dimension,
    along with a CAM heatmap.

    Parameters
    ----------
    sample : np.ndarray
        Original multivariate time series sample of shape (timesteps, channels).
    counterfactuals_dict : dict
        Dictionary where keys are method names and values are counterfactual arrays of the same shape as sample.
    cam : np.ndarray
        Class Activation Map (CAM) for the given instance, shape (timesteps,).
    dimension_idx : int
        Index of the dimension/channel to plot.
    title : str, optional
        Title for the plot (e.g., "Dimension 4").
    """

    sample_reshaped = sample.T
    timesteps = np.arange(sample.shape[0])

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6), sharex=True, gridspec_kw={'height_ratios': [3, 0.2]})

    # Original time series
    ax1.plot(timesteps, sample_reshaped[dimension_idx], label="Original Time Series", color='#B0B1B2', linewidth=2)

    # Plot counterfactuals only where they differ
    linestyles = ['--', ':', '-.', (0, (3, 1, 1, 1))]
    colors = ['#ff595a', '#457b9d', '#ffb703', '#6E2594']
    for (method, cf_array), linestyle, color in zip(counterfactuals_dict.items(), linestyles, colors):
        cf_reshaped = cf_array.T
        diffs = np.where(sample_reshaped[dimension_idx] != cf_reshaped[dimension_idx])[0]
        if diffs.size > 0:
            start_diff, end_diff = diffs[0], diffs[-1]
            ax1.plot(range(start_diff, end_diff + 1), cf_reshaped[dimension_idx, start_diff:end_diff + 1],
                     label=method, linestyle=linestyle, color=color, linewidth=2)
            if start_diff > 0:
                ax1.plot([start_diff - 1, start_diff],
                         [sample_reshaped[dimension_idx, start_diff - 1], cf_reshaped[dimension_idx, start_diff]],
                         color=color, linewidth=2)
            if end_diff < sample_reshaped[dimension_idx].size - 1:
                ax1.plot([end_diff, end_diff + 1],
                         [cf_reshaped[dimension_idx, end_diff], sample_reshaped[dimension_idx, end_diff + 1]],
                         color=color, linewidth=2)

    ax1.set_title(title or f"Dimension {dimension_idx + 1}",fontsize=20)
    ax1.set_ylabel("Value", fontsize=17)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.grid(True)

    # Normalize CAM
    cam_normalized = np.interp(cam, (cam.min(), cam.max()), (0, 1))
    ax2.imshow(cam_normalized[np.newaxis, :], aspect='auto', cmap='YlOrRd', extent=[0, len(timesteps), 0, 1])
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["CAM"], fontsize=16, rotation='vertical')
    ax2.set_xlabel("Time", fontsize=17)
    ax2.grid(False)

    plt.tight_layout()

    # Save as PDF (optional)
    if title:
        filename = f"{title.replace(' ', '_').lower()}.pdf"
    else:
        filename = f"dimension_{dimension_idx + 1}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

def plot_method_comparison_separated_with_cam(sample: np.ndarray,
                                           counterfactuals: dict,
                                           weights: np.ndarray,
                                           dimension: int = 0) -> None:
    """
    Plots counterfactuals from 3 methods (CONFETTI, CoMTE, TSEvo) in horizontal subplots.
    Only the first subplot (CONFETTI) displays the CAM heatmap.
    Ensures tight x-axis alignment and shared y-axis across all plots.

    Parameters:
    - sample (np.ndarray): Original time series, shape (timesteps, features).
    - counterfactuals (dict): Dictionary of counterfactual arrays keyed by method name.
    - weights (np.ndarray): Class Activation Map weights (1D).
    - dimension (int): Dimension index to visualize (default: 0).
    """
    methods = ["CONFETTI", "CoMTE", "TSEvo"]
    cf_colors = {
        "CONFETTI": "#ff595a",
        "CoMTE": "#2660A4",
        "TSEvo": "#F3B700"
    }

    sample_dim = sample.T[dimension]
    cf_dims = [cf.T[dimension] for cf in counterfactuals.values()]
    y_min = min([sample_dim.min()] + [cf.min() for cf in cf_dims])
    y_max = max([sample_dim.max()] + [cf.max() for cf in cf_dims])
    weights_norm = np.interp(weights, (weights.min(), weights.max()), (0, 1))

    timesteps = np.arange(len(sample_dim))

    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3, 0.2])

    original_color = '#64cdc0'

    for i, method in enumerate(methods):
        cf_dim = counterfactuals[method].T[dimension]
        cf_color = cf_colors[method]

        ax_ts = fig.add_subplot(gs[0, i])
        ax_ts.plot(timesteps, sample_dim, label='Original Time Series', color=original_color, linewidth=2)

        diffs = np.where(sample_dim != cf_dim)[0]
        if diffs.size > 0:
            start_diff, end_diff = diffs[0], diffs[-1]
            ax_ts.plot(timesteps[start_diff:end_diff + 1],
                       cf_dim[start_diff:end_diff + 1],
                       label='Counterfactual', color=cf_color, linewidth=2, linestyle='--')

            if start_diff > 0:
                ax_ts.plot([timesteps[start_diff - 1], timesteps[start_diff]],
                           [sample_dim[start_diff - 1], cf_dim[start_diff]],
                           color=cf_color, linewidth=2)
            if end_diff < len(sample_dim) - 1:
                ax_ts.plot([timesteps[end_diff], timesteps[end_diff + 1]],
                           [cf_dim[end_diff], sample_dim[end_diff + 1]],
                           color=cf_color, linewidth=2)

        ax_ts.set_xlim(0, len(sample_dim) - 1)
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.set_title(f"{method}", fontsize=16)
        ax_ts.set_ylabel("Value", fontsize=15)
        ax_ts.tick_params(axis='both', labelsize=11)
        ax_ts.legend(fontsize=12, loc='upper right')
        ax_ts.grid(True)

        if i == 0:
            ax_cam = fig.add_subplot(gs[1, i], sharex=ax_ts)
            ax_cam.imshow(weights_norm[np.newaxis, :], aspect="auto", cmap="YlOrRd")
            ax_cam.set_xlim(0, len(sample_dim) - 1)
            ax_cam.set_yticks([])
            ax_cam.set_ylabel("CAM", fontsize=12)
            ax_cam.set_xlabel("Time", fontsize=15)
            ax_cam.tick_params(axis='x', labelsize=11)
            ax_cam.grid(False)
        else:
            fig.add_subplot(gs[1, i]).axis("off")

    plt.tight_layout()
    plt.show()


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