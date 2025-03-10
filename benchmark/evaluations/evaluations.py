import pandas as pd
from evaluator import Evaluator

# Initialize Evaluators
confetti_ev = Evaluator('confetti_naive')
comte_ev = Evaluator('comte')
sets_ev = Evaluator('sets')
tsevo_ev = Evaluator('tsevo')

# Evaluate datasets
confetti_bm, confetti_bm_summary = confetti_ev.evaluate_dataset('BasicMotions')
comte_bm, comte_bm_summary = comte_ev.evaluate_dataset('BasicMotions')
sets_bm, sets_bm_summary = sets_ev.evaluate_dataset('BasicMotions')
tsevo_bm, tsevo_bm_summary = tsevo_ev.evaluate_dataset('BasicMotions')

# Add "Explainer" column
confetti_bm_summary["Explainer"] = "Confetti"
comte_bm_summary["Explainer"] = "Comte"
sets_bm_summary["Explainer"] = "Sets"
tsevo_bm_summary["Explainer"] = "TSEvo"

# Concatenate all dataframes
final_df = pd.concat([confetti_bm_summary, comte_bm_summary, sets_bm_summary, tsevo_bm_summary], ignore_index=True)

# Display final dataframe
print(final_df)
