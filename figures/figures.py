import pandas as pd
import config as cfg
from pathlib import Path
from confetti.explainer.utils import load_data, load_multivariate_ts_from_csv, convert_string_to_array
from confetti.CAM import compute_weights_cam
import keras

results = pd.read_csv(Path.cwd().parent/'benchmark'/'evaluations'/'all_evaluation_results.csv')
column_order = ['Explainer', 'Model', 'Dataset', 'Alpha', 'Param Config', 'Coverage', 'Validity', 'Confidence',
                    'Sparsity', 'Proximity L1', 'Proximity L2', 'Proximity DTW', 'yNN']
results = results[column_order]

dataset = "BasicMotions"
model_name = "fcn"
counterfactuals_path = Path.cwd().parent / "results" / dataset

#Load results
confetti_fcn_bm = pd.read_csv(counterfactuals_path / f"confetti_optimized_{dataset}_{model_name}_theta_0.85.csv")
comte_fcn_bm =  pd.read_csv(counterfactuals_path / f"comte_{dataset}_{model_name}_counterfactuals.csv")
sets_fcn_bm = pd.read_csv(counterfactuals_path/ f"sets_{dataset}_{model_name}_counterfactuals.csv")
tsevo_fcn_bm = pd.read_csv(counterfactuals_path / f"tsevo_{dataset}_{model_name}_counterfactuals.csv")

# Transform results into arrays
confetti_fcn_bm['Solution'] = confetti_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
comte_fcn_bm['Solution'] = comte_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
sets_fcn_bm['Solution'] = sets_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
tsevo_fcn_bm['Solution'] = tsevo_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))

#Select one instance to display
confetti_ce = confetti_fcn_bm.iloc[2]['Solution']
comte_ce = comte_fcn_bm.iloc[2]['Solution']
sets_ce = sets_fcn_bm.iloc[2]['Solution']
tsevo_ce = tsevo_fcn_bm.iloc[2]['Solution']

counterfactuals_dict = {
    'CONFETTI': confetti_ce,
    'CoMTE': comte_ce,
    'TSEvo': tsevo_ce,
    'SETS': sets_ce
}



dataset = "BasicMotions"
model_name = "fcn"
counterfactuals_path = Path.cwd().parent / "results" / dataset

#Load results
confetti_fcn_bm = pd.read_csv(counterfactuals_path / f"confetti_optimized_{dataset}_{model_name}_theta_0.85.csv")
comte_fcn_bm =  pd.read_csv(counterfactuals_path / f"comte_{dataset}_{model_name}_counterfactuals.csv")
sets_fcn_bm = pd.read_csv(counterfactuals_path/ f"sets_{dataset}_{model_name}_counterfactuals.csv")
tsevo_fcn_bm = pd.read_csv(counterfactuals_path / f"tsevo_{dataset}_{model_name}_counterfactuals.csv")

# Transform results into arrays
confetti_fcn_bm['Solution'] = confetti_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
comte_fcn_bm['Solution'] = comte_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
sets_fcn_bm['Solution'] = sets_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))
tsevo_fcn_bm['Solution'] = tsevo_fcn_bm['Solution'].apply(lambda x: convert_string_to_array(x, timesteps=100, channels=6))

#Select one instance to display
confetti_ce = confetti_fcn_bm.iloc[2]['Solution']
comte_ce = comte_fcn_bm.iloc[2]['Solution']
sets_ce = sets_fcn_bm.iloc[2]['Solution']
tsevo_ce = tsevo_fcn_bm.iloc[2]['Solution']

counterfactuals_dict = {
    'CONFETTI': confetti_ce,
    'CoMTE': comte_ce,
    'TSEvo': tsevo_ce,
    'SETS': sets_ce
}

# Load model
model_path = Path.cwd().parent / "models" / "trained_models" / dataset / f"{dataset}_{model_name}.keras"
model = keras.models.load_model(model_path)

# Load training dataset
X_train, _, _, _ = load_data(dataset, one_hot=False)

#Compute training weights
weights = compute_weights_cam(model=model, X_data=X_train, dataset=dataset, save_weights=False, data_type='training')
nun_weight = weights[27]

#load samples
sample_file = f"{cfg.DATA_DIR}/{dataset}_{model_name}_samples.csv"
X_samples, y_samples = load_multivariate_ts_from_csv(sample_file)

sample = X_samples[0]


## Figure 1. Comparison of Counterfactual Explanations for MTS by different methods.

from visualizer import plot_method_comparison_with_cam

plot_method_comparison_with_cam(sample, counterfactuals_dict, nun_weight, dimension_idx=3,
                                title="Counterfactual Comparison for a Single Dimension (4th Dimension)")

## Figure 2. Sensitivity Analysis
# Obtain all the rows where the Explainer contains somehow 'confetti'
confetti_results = results[results['Explainer'].str.contains('Confetti Optimized', case=False, na=False)]
results_alphas = confetti_results[confetti_results['Alpha'] == True]
results_thetas = confetti_results[confetti_results['Alpha'] == False]

from visualizer import boxplot_all_tradeoffs_by_model
boxplot_all_tradeoffs_by_model(results_alphas, results_thetas)
