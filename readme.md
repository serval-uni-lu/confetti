<!-- Build Status -->
  <a href="https://github.com/serval-uni-lu/confetti/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/serval-uni-lu/confetti/ci.yaml?label=CI&logo=github" alt="CI Status">
  </a>

<!-- Documentation -->
  <a href="https://confetti-ts.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/confetti-ts?logo=readthedocs" alt="Documentation Status">
  </a>

<!-- License -->
  <a href="https://github.com/serval-uni-lu/confetti/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/serval-uni-lu/confetti?color=4E65FF" alt="License">
  </a>


![CONFETTI Logo](https://raw.githubusercontent.com/serval-uni-lu/confetti/main/docs/artwork/confetti-logo.png)

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

### PyPI Installation
```bash
pip install confetti-ts
```

### Development Installation
```bash
git clone https://github.com/serval-uni-lu/confetti.git
cd confetti

uv venv
source .venv/bin/activate
uv pip install -e .
```

Requirements:

* Python 3.12+
* NumPy, pandas
* Keras 3.x
* TensorFlow 
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

![Counterfactual Example](https://raw.githubusercontent.com/serval-uni-lu/confetti/main/docs/artwork/counterfactual_example.png)

In the visualization:

* green curves represent the original instance
* red curves represent the counterfactual subsequence
* the heatmap corresponds to CAM scores of the nearest unlike neighbor

The alignment between CAM activation and the altered subsequence shows how CONFETTI uses attribution to target meaningful areas of the time series.

---
## 📚Documentation

The full documentation, including usage guides, API reference, and examples, is available at:

👉 **https://confetti-ts.readthedocs.io/en/latest/**

---
## 📄License

CONFETTI is released under the [MIT License](LICENSE). 

---
## 📝 Citing CONFETTI
A formal citation entry will appear here once the paper is officially published.

To **replicate the experiments described in the paper**, use the **`paper` branch** of this
repository. It contains the experiment scripts, model configurations, and dataset handling
used in the publication.



