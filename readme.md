

<p align="center">
  <img src="docs/artwork/confetti-logo.svg" width="600"  alt="CONFETTI logo">
</p>

---
# Counterfactual Explanations for Multivariate Time Series



**CONFETTI** is a multi-objective method for generating **counterfactual explanations for multivariate time series**. 
It identifies the most influential subsequences, constructs a minimal perturbation, and optimizes it under multiple objectives to produce **sparse**, **realistic**, and **confidence-increasing** counterfactuals 

CONFETTI is model-agnostic and works with **any deep learning classifier**, differentiable or not.

--- 
## ✨ Highlights

* Multi-objective optimization using NSGA-III
* Works for any **Keras/Scikit-learn** multivariate time series classifier
* Optional use of **class activation maps** for feature-weighted perturbations
* Generates multiple diverse counterfactuals per instance
* Parallelized counterfactual generation
* Built-in utilities for:
  * loading datasets
  * computing CAM weights
  * visualizing counterfactual explanations

---
## 🚀 Installation

### Development Installation
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### PyPI (once released)
```bash
pip install confetti-ts
```
Requirements:

* Python 3.12+
* NumPy, pandas
* Keras 3.x
* Pymoo
* tslearn

All dependencies are handled automatically via ``pyproject.toml``.

---

## ⚡ Quick Example
Below is a minimal end-to-end example based on the ``demo_confetti.ipynb`` notebook.
It loads a trained model, prepares a dataset, and generates counterfactuals for a single instance.

```python
from confetti import CONFETTI
from confetti.attribution import cam
from confetti.utils import load_multivariate_ts_from_csv
from confetti.visualizations import plot_counterfactual
import keras

# Load model
model_path = "examples/models/toy_fcn.keras"
model = keras.models.load_model(model_path)

# Load dataset in (n_samples, time_steps, n_features) format
X_train, y_train = load_multivariate_ts_from_csv("examples/data/toy_train.csv")
X_test, y_test   = load_multivariate_ts_from_csv("examples/data/toy_test.csv")

# Select instance to explain
instance = X_test[0:1]

# Generate CAM weights for training data (optional)
training_weights = cam(model, X_train)

# Initialize explainer
explainer = CONFETTI(model_path=model_path)

# Generate counterfactuals
results = explainer.generate_counterfactuals(
    instances_to_explain=instance,
    reference_data=X_train,
    reference_weights=training_weights,      # or None if not available
)

# Visualize the best counterfactual
plot_counterfactual(
    original=results[0].original_instance,
    counterfactual=results[0].best,
    cam_weights=results[0].feature_importance,
    cam_mode="heatmap",
    title="Counterfactual Explanation"
)
```
<p> <img src="docs/artwork/counterfactual_example.png" alt="Counterfactual Example" width="650"/> </p>

In the visualization:

* green curves represent the original instance
* red curves represent the counterfactual subsequence
* the heatmap corresponds to CAM scores of the nearest unlike neighbor

The alignment between CAM activation and the altered subsequence shows how CONFETTI uses attribution to target meaningful areas of the time series.

---
## 📚Documentation
Documentation is currently in preparation and will be available soon.

---
## 📄License
To be added before release.

---
## 📝 Citing CONFETTI
A formal citation entry will appear here once the paper is officially released.




