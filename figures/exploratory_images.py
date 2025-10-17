from visualizer import plot_counterfactual_with_cam
from confetti.explainer.utils import convert_string_to_array
import pandas as pd
from pathlib import Path


def exploratory_images():
    dataset = "BasicMotions"
    model_name = "fcn"
    counterfactuals_path = Path.cwd().parent / "benchmark" / "experiments" / "proximity_metric_results"

    # Load results
    euclidean = pd.read_csv(counterfactuals_path / f"{dataset}_euclidean_optimized.csv")
    manhattan = pd.read_csv(counterfactuals_path / f"{dataset}_manhattan_optimized.csv")
    dtw = pd.read_csv(counterfactuals_path / f"{dataset}_dtw_optimized.csv")

    # Transform results into arrays
    euclidean['Solution'] = euclidean['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
    manhattan['Solution'] = manhattan['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
    dtw['Solution'] = dtw['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))


    # Select one instance to display
    euclidean_ce = euclidean.iloc[0]['Solution']
    manhattan_ce = manhattan.iloc[0]['Solution']
    dtw_ce = dtw.iloc[0]['Solution']

    counterfactuals_dict = {
        'Euclidean': euclidean_ce,
        'Manhattan': manhattan_ce,
        'DTW': dtw_ce
    }

    plot_counterfactual_with_cam(

    )