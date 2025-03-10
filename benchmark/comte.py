import numpy as np
import keras
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
import pandas as pd
from pathlib import Path
from confetti.explainer.utils import load_data
import config as cfg

# Define the datasets to be evaluated
datasets = ['BasicMotions']

for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(dataset, encode_labels=True)

    # Load the pre-trained Fully Convolutional Network (FCN) model
    model_path=cfg.TRAINED_MODELS_DIR / dataset / f'{dataset}_fcn.keras'
    model = keras.models.load_model(str(model_path))
    print("Model loaded.")

    # Initialize the CoMTE counterfactual explainer
    exp_model = COMTECF(model,(X_train,y_train),mode='time', backend='TF', method= 'brute')
    results_comte = pd.DataFrame(columns=["Solution", "Test Instance", "Original Label", "CE Label"])
    print("Explainer created.")

    # Generate counterfactual explanations for each test instance
    for i, instance in enumerate(X_test):
        # Reshape the test instance to match the model's expected input format
        item = X_test[i].reshape(1, *X_test[i].shape)
        y_target = model.predict(item)

        # Generate the counterfactual explanation using CoMTE
        exp = exp_model.explain(item)
        array, label = exp

        # Extract original and counterfactual labels
        org_label = np.argmax(y_target)
        cf_label = label[0]
        exp = np.array(array)  # Ensure exp is a NumPy array

        # Store results in a DataFrame
        results_dict = {'Solution': [exp[0]], 'Test Instance': i, 'Original Label': org_label, 'CE Label':cf_label}
        results_df = pd.DataFrame(results_dict)
        results_comte = pd.concat([results_comte, results_df], ignore_index=True)

    # Save the results as a CSV file
    results_directory = cfg.RESULTS_DIR / dataset
    results_comte.to_csv(results_directory / 'comte_counterfactuals.csv', index=False)







