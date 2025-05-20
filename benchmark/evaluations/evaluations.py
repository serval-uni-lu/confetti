import pandas as pd
from evaluator import Evaluator

# Initialize Evaluator
ev = Evaluator()

# Evaluate datasets
comte_bm, comte_bm_summary = ev.evaluate_dataset('comte', 'BasicMotions', 'fcn')
sets_bm, sets_bm_summary = ev.evaluate_dataset('sets', 'BasicMotions', 'fcn')
tsevo_bm, tsevo_bm_summary = ev.evaluate_dataset('tsevo', 'BasicMotions', 'fcn')
naive_bm, naive_bm_summary = ev.evaluate_dataset('confetti_naive', 'BasicMotions',
                                                 'fcn', alpha=True, param_config=0.1)
optimized_bm, optimized_bm_summary = ev.evaluate_dataset('confetti_optimized', 'BasicMotions',
                                                            'fcn', alpha=True, param_config=0.1)

# Add "Explainer" column
comte_bm_summary["Explainer"] = "Comte"
sets_bm_summary["Explainer"] = "Sets"
tsevo_bm_summary["Explainer"] = "TSEvo"
naive_bm_summary["Explainer"] = "Confetti Naive"
optimized_bm_summary["Explainer"] = "Confetti Optimized"

# Concatenate all dataframes
final_df = pd.concat([comte_bm_summary, sets_bm_summary, tsevo_bm_summary, naive_bm_summary,optimized_bm_summary], ignore_index=True)

# Display final dataframe
print(final_df)
