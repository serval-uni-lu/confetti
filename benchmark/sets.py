from confetti.explainer.utils import load_data
from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF
import keras
import numpy as np
import pandas as pd
from pathlib import Path
import config as cfg

# Define the dataset to be evaluated
datasets = ["BasicMotions"]

for dataset in datasets:
    # Load the dataset
    X_train, X_test, y_train, y_test = load_data(dataset, encode_labels=False)

    # Load the pre-trained Fully Convolutional Network (FCN) model
    model_path = cfg.TRAINED_MODELS_DIR / dataset /f'{dataset}_fcn.keras'
    model = keras.models.load_model(model_path)

    # Explainer
    exp_model = SETSCF(model,
                       (X_train, y_train),
                       backend='TF',
                       mode='time',
                       min_shapelet_len=3,
                       max_shapelet_len=20,
                       time_contract_in_mins_per_dim=1,
                       fit_shapelets=False)

    exp_model.fit(occlusion_threshhold=1e-1, remove_multiclass_shapelets=True)
    print("Explanation model fitted.")

    # Create DataFrame to store results
    results_df = pd.DataFrame(columns=["Solution", "Test Instance", "Original Label", "CE Label"])

    # Iterate through all X_test instances
    for i in range(len(X_test)):
        item = X_test[i]
        cf_explanation, label_cf = exp_model.explain(item, target=None)

        # Add results to DataFrame
        row_dict = {'Solution': [cf_explanation], 'Test Instance': i, 'Original Label': y_test[i], 'CE Label': label_cf}
        row_df = pd.DataFrame(row_dict)
        results_df = pd.concat([results_df, row_df], ignore_index=True)

    # Save results as CSV
    results_directory = cfg.RESULTS_DIR / dataset
    results_directory.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_directory / 'sets_counterfactuals.csv', index=False)

    print(f"Results saved for {dataset} at {results_directory}/sets_counterfactuals.csv")
