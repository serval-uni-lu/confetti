import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_counterfactual_comparison(naive_csv, optimized_csv, figsize=(20, 20), constrained_layout=True):
    """
    Plot a comparison of naive and optimized counterfactual solutions based on Sparsity and Precision.

    Parameters:
    - naive_csv (str): Path to the CSV file containing naive counterfactuals.
    - optimized_csv (str): Path to the CSV file containing optimized counterfactuals.
    - figsize (tuple): The size of the plot. Default is (20, 20).
    - constrained_layout (bool): Whether to apply constrained layout. Default is True.
    """
    # Read CSV files into DataFrames
    df_naive = pd.read_csv(naive_csv)
    df_optimized = pd.read_csv(optimized_csv)

    # Define colors
    naive_color = '#ff595a'
    optimized_color = '#018575'

    # Group by 'Test Instance' for both dataframes
    grouped_original = df_optimized.groupby('Test Instance')
    grouped_naive = df_naive.groupby('Test Instance')

    # Plotting
    fig, axs = plt.subplots(8, 5, figsize=figsize, constrained_layout=constrained_layout)
    axs = axs.ravel()

    # Loop through test instances and plot data
    for i, test_instance in enumerate(grouped_original.groups.keys()):
        ax = axs[i]
        group_original = grouped_original.get_group(test_instance)
        group_naive = grouped_naive.get_group(test_instance)
        ax.scatter(group_original['Sparsity'], group_original['Precision'], s=60, color=optimized_color, label='Optimized')
        ax.scatter(group_naive['Sparsity'], group_naive['Precision'], s=60, marker="s", color=naive_color, label='Naive')
        ax.set_title(f'Instance: {test_instance}')
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Confidence')
        if i == 0:
            ax.legend()

    # Show plot
    plt.show()

def plot_mean_sparsity_confidence(results_df, figsize=(14, 6), naive_width=0.55, optimized_width=0.33):
    """
    Plot mean sparsity and mean confidence for naive and optimized counterfactuals.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the dataset names and metrics.
    - figsize (tuple): Figure size. Default is (14, 6).
    - naive_width (float): Width of the bars for naive counterfactuals. Default is 0.55.
    - optimized_width (float): Width of the bars for optimized counterfactuals. Default is 0.33.
    """
    # Define colors
    naive_color = '#ff595a'
    optimized_color = '#018575'

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot Mean Sparsity
    axes[0].bar(results_df['Dataset'], results_df['Sparsity Naive'], width=naive_width, label='Naive', align='center', color=naive_color, hatch="//")
    axes[0].bar(results_df['Dataset'], results_df['Sparsity Optimized'], width=optimized_width, label='Optimized', align='edge', color=optimized_color)
    axes[0].set_title('Mean Sparsity')
    axes[0].set_ylabel('Sparsity')

    # Plot Mean Confidence
    axes[1].bar(results_df['Dataset'], results_df['Confidence Naive'], width=naive_width, label='Naive', align='center', color=naive_color, hatch="//")
    axes[1].bar(results_df['Dataset'], results_df['Confidence Optimized'], width=optimized_width, label='Optimized', align='edge', color=optimized_color)
    axes[1].set_title('Mean Confidence')
    axes[1].set_ylabel('Confidence')

    # Set x-labels with rotation
    for ax in axes:
        ax.set_xlabel('Dataset')
        ax.set_xticklabels(results_df['Dataset'], rotation=45, ha='right')

    # Adjust legend position
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def visualize_time_series(series, num_dimensions_to_show=3, line_color='#FF595A', line_width=2):
    """
    Visualize a multivariate time series by plotting a specified number of dimensions.

    Parameters:
    - series (np.ndarray or pd.DataFrame): Multivariate time series data (shape: time_steps x dimensions).
    - num_dimensions_to_show (int): Number of dimensions to visualize. Default is 3.
    - line_color (str): Color of the plotted lines. Default is '#FF595A'.
    - line_width (int): Width of the plotted lines. Default is 2.
    """
    # Transpose if needed (assumes input shape: (time_steps, dimensions))
    series_reshaped = np.asarray(series).T

    # Determine the total number of dimensions
    total_dimensions = series_reshaped.shape[0]

    # Limit the number of dimensions to show
    num_dimensions = min(num_dimensions_to_show, total_dimensions)

    # Set the aesthetic style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create subplots
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions), sharex=True)

    # Ensure axes is iterable for single-dimension case
    if num_dimensions == 1:
        axes = [axes]

    # Plot each selected dimension
    for i, ax in enumerate(axes):
        ax.plot(series_reshaped[i], color=line_color, linewidth=line_width, label=f"Dimension {i+1}")
        ax.set_title(f"Dimension {i+1}", fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)

    # Improve layout
    plt.tight_layout()
    plt.show()

def visualize_counterfactuals(ce, instance: int, optimized=False, precision=True):
    sample = ce.X_test[instance]
    if not optimized:
        counterfactual = ce.get_naive_counterfactual(instance)
    else:
        counterfactual = ce.get_optimized_counterfactual(instance, precision)

    # Reshape Time Series for consistency with usual time series format [time, dimension]
    sample_reshaped = sample.T
    counterfactual_reshaped = counterfactual.T

    # Determine the number of dimensions (subplots) to visualize (limited to 3)
    num_dimensions = min(3, sample_reshaped.shape[0])

    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create a plot with a dynamic number of subplots for the time series
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions))

    # Iterate through each dimension up to the first three
    for i in range(num_dimensions):
        ax = axes[i] if num_dimensions > 1 else axes  # Handle case with a single dimension plot

        # Plot the original series
        original_line, = ax.plot(sample_reshaped[i], label='Original', color='#64cdc0', linewidth=2)

        # Find the indices where the original and counterfactual differ
        diffs = np.where(sample_reshaped[i] != counterfactual_reshaped[i])[0]

        if diffs.size > 0:  # If there are differences
            # Determine the start and end of the continuous differing sequence
            start_diff = diffs[0]
            end_diff = diffs[-1]

            # Plot the entire sub-sequence where differences occur in salmon color
            counterfactual_line, = ax.plot(range(start_diff, end_diff + 1),
                                           counterfactual_reshaped[i, start_diff:end_diff + 1],
                                           label='Counterfactual', color='#ff595a', linewidth=2, linestyle='--')

            # Connecting line: from the last unchanged point to the first changed point
            start_diff = diffs[0]  # Start of the differing subsequence
            if start_diff > 0:  # Make sure there's a point to connect from
                ax.plot([start_diff - 1, start_diff],
                        [sample_reshaped[i, start_diff - 1], counterfactual_reshaped[i, start_diff]], color='salmon',
                        linewidth=2)

            # Connecting line: from the last changed point to the next original point
            end_diff = diffs[-1]  # End of the differing subsequence
            if end_diff < sample_reshaped[i].size - 1:  # Make sure there's a point to connect to
                ax.plot([end_diff, end_diff + 1],
                        [counterfactual_reshaped[i, end_diff], sample_reshaped[i, end_diff + 1]], color='salmon',
                        linewidth=2)

        # Set legend manually to ensure no duplicate labels
        if diffs.size > 0:
            ax.legend(handles=[original_line, counterfactual_line], fontsize=12)
        else:
            ax.legend(handles=[original_line], fontsize=12)

        ax.set_title(f"Dimension {i + 1}", fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True)  # Add gridlines

    plt.tight_layout()
    plt.show()

def plot_with_cam_as_line(ce, instance: int, optimized=False, precision=True, weights=None):
    """
    Visualizes the counterfactual time series against the original series for a given instance including its CAM.

    Parameters:
    - ce: An object containing the dataset and methods to retrieve counterfactuals.
    - instance (int): Index of the instance to visualize.
    - optimized (bool): Whether to use the optimized counterfactual (default: False).
    - precision (bool): If optimized=True, whether to optimize for precision (default: True).
    - weights (np.ndarray or None): Class Activation Map (CAM) weights (default: None).

    Returns:
    - pd.DataFrame: A DataFrame containing original, counterfactual, and optional CAM weights for visualization.
    """
    sample = ce.X_test[instance]
    counterfactual = ce.get_optimized_counterfactual(instance, precision) if optimized else ce.get_naive_counterfactual(
        instance)

    # Reshape for time-series format: [time, dimension]
    sample_reshaped, counterfactual_reshaped = sample.T, counterfactual.T

    # Limit visualization to 3 dimensions
    num_dimensions = min(3, sample_reshaped.shape[0])

    # Normalize CAM weights to (-15, 15) if provided
    if weights is not None:
        weights = np.interp(weights, (weights.min(), weights.max()), (-15, 15))

    # Plot setup
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions), sharex=True)

    # Ensure axes is iterable when num_dimensions = 1
    if num_dimensions == 1:
        axes = [axes]

    # Define colors
    original_color, cf_color, cam_color = '#64cdc0', '#ff595a', '#ffa500'

    # Plot each dimension
    for i, ax in enumerate(axes):
        # Plot original series
        ax.plot(sample_reshaped[i], label='Original Time Series', color=original_color, linewidth=2)

        # Find differing indices
        diffs = np.where(sample_reshaped[i] != counterfactual_reshaped[i])[0]

        if diffs.size > 0:
            start_diff, end_diff = diffs[0], diffs[-1]

            # Plot counterfactual segment
            ax.plot(range(start_diff, end_diff + 1), counterfactual_reshaped[i, start_diff:end_diff + 1],
                    label='Counterfactual Segment', color=cf_color, linewidth=2, linestyle='--')

            # Connecting lines at transition points
            if start_diff > 0:
                ax.plot([start_diff - 1, start_diff],
                        [sample_reshaped[i, start_diff - 1], counterfactual_reshaped[i, start_diff]],
                        color=cf_color, linewidth=2)
            if end_diff < sample_reshaped[i].size - 1:
                ax.plot([end_diff, end_diff + 1],
                        [counterfactual_reshaped[i, end_diff], sample_reshaped[i, end_diff + 1]],
                        color=cf_color, linewidth=2)

        # Plot CAM weights if provided
        if weights is not None:
            ax.plot(weights, label='CAM Weights', color=cam_color, linewidth=2, linestyle=':')

        # Set title, labels, and legend
        ax.set_title(f"Dimension {i + 1}", fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True)

        # Ensure appropriate legend entries
        legend_handles = [ax.lines[0]]  # Always include original series
        if diffs.size > 0:
            legend_handles.append(ax.lines[1])  # Counterfactual segment
        if weights is not None:
            legend_handles.append(ax.lines[-1])  # CAM weights
        ax.legend(handles=legend_handles, fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Prepare DataFrame output
    data_dict = {f'Original_Dim_{i + 1}': sample_reshaped[i] for i in range(num_dimensions)}
    data_dict.update({f'Counterfactual_Dim_{i + 1}': counterfactual_reshaped[i] for i in range(num_dimensions)})
    if weights is not None:
        data_dict['CAM_Weights'] = weights

    return pd.DataFrame(data_dict)

def plot_with_cam_as_heatmap(sample: np.ndarray,
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


def plot_method_comparison_with_cam_horizontal(sample: np.ndarray,
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



def tradeoff_sparsity_proximity(df):
    """
    Plots each row from the dataframe in a space where X is df['Proximity'] and Y is df['Sparsity'].

    Parameters:
    df (pd.DataFrame): A dataframe containing 'Proximity' and 'Sparsity' columns.
    """
    if 'Proximity' not in df.columns or 'Sparsity' not in df.columns:
        raise ValueError("DataFrame must contain 'Proximity' and 'Sparsity' columns.")

    plt.figure(figsize=(8, 6))
    plt.scatter(df['Proximity'], df['Sparsity'], color='#64cdc0', alpha=0.7, label='Samples')

    plt.xlabel("Proximity")
    plt.ylabel("Sparsity")
    plt.title("Trade-off between Sparsity and Proximity")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()