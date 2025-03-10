import numpy as np
import keras
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import pandas as pd
from pathlib import Path
from confetti.explainer.utils import load_data
import config as cfg

# Datasets to process
datasets = ['BasicMotions']

for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(dataset, encode_labels=True)

    # Load corresponding FCN model
    model_path = cfg.TRAINED_MODELS_DIR / dataset /f'{dataset}_fcn.keras'
    model = keras.models.load_model(str(model_path))
    # Create explainer
    exp = TSEvo(model=model, data=(X_test, y_test), backend='TF', mode='time', epochs=2, verbose=0)
    # DataFrame where results will be stored
    results_df = pd.DataFrame(columns=["Solution", "Test Instance", "Original Label", "CE Label"])

    for i in range(len(X_test)):
        item = X_test[i].reshape(1, *X_test[i].shape)
        label = np.array([np.argmax(y_test[i])])
        pop, label_cf = exp.explain(item, label)

        row_dict = {'Solution': [pop], 'Test Instance': i, 'Original Label': label[0], 'CE Label': np.argmax(label_cf)}
        row_df = pd.DataFrame(row_dict)
        # Add results to dataframe
        results_df = pd.concat([results_df,row_df],ignore_index=True)
    # Save results as CSV
    results_directory = cfg.RESULTS_DIR / dataset
    results_directory.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_directory / 'tsevo_counterfactuals.csv', index=False)

    print(f"Results saved for {dataset} at {results_directory}/tsevo_counterfactuals.csv")