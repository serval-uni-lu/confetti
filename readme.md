<p align="center">
<!-- PyPi Version -->
  <a href="https://pypi.org/project/confetti-ts/">
    <img src="https://img.shields.io/pypi/v/confetti-ts?logo=pypi&logoColor=white" alt="PyPI Version">
  </a>
  
  <!-- Python Versions -->
  <a href="https://pypi.org/project/confetti-ts/">
    <img src="https://img.shields.io/pypi/pyversions/confetti-ts?logo=python&logoColor=white" alt="Python Versions">
  </a>
  
  <!-- Wheel -->
  <img src="https://img.shields.io/pypi/wheel/confetti-ts" alt="Wheel">
  
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

</p>

![CONFETTI Logo](https://raw.githubusercontent.com/serval-uni-lu/confetti/main/docs/artwork/confetti-logo.png)

---
# Counterfactual Explanations for Time Series and Tabular Data



**CONFETTI** is a multi-objective method for generating **counterfactual explanations** for **multivariate time series** and **tabular data** classifiers.
It identifies the most influential features or temporal regions, constructs a minimal perturbation, and optimizes it under multiple objectives to produce **sparse**, **realistic**, and **confidence-increasing** counterfactuals.

CONFETTI is model-agnostic and works with **any classifier** — Keras, PyTorch, or scikit-learn.

--- 
## ✨ Highlights

* Multi-objective optimization using **NSGA-III**
* **Time series**: works with any Keras/scikit-learn multivariate time series classifier
* **Tabular data**: works with any classifier exposing `predict_proba` or `predict`
* **Relation constraints for tabular data**: encode domain rules on counterfactuals (e.g. `age <= retirement_age`)
* **Rust-accelerated** backend for distances, NSGA-III, and constraint evaluation
* Optional use of **class activation maps** for feature-weighted perturbations (time series)
* Built-in distance metrics: Euclidean, Manhattan, DTW, Soft-DTW, GAK, CTW, Gower
* Support for **categorical features**, **immutable features**, and **Gower distance** for mixed-type data
* Generates multiple diverse counterfactuals per instance
* Parallelized counterfactual generation

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

Core dependencies:

* Python 3.12+
* NumPy, pandas, scikit-learn, joblib

Optional:

* Keras 3.x + TensorFlow (`pip install confetti-ts[keras]`)
* PyTorch (`pip install confetti-ts[torch]`)

All dependencies are handled automatically via ``pyproject.toml``.

---

## ⚡ Quick Example — Time Series

Below is a minimal end-to-end example for multivariate time series counterfactuals.
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

---

## ⚡ Quick Example — Tabular Data

Generate counterfactual explanations for tabular classifiers using `TabularCONFETTI`.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from confetti import TabularCONFETTI, Feature

# Prepare data
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize explainer
explainer = TabularCONFETTI(model)

# Generate counterfactuals
results = explainer.generate_counterfactuals(
    instances_to_explain=X.iloc[:5],
    reference_data=X,
    alpha=0.5,
    theta=0.51,
)

# Inspect the best counterfactual
cf_set = results[0]
print("Original label:     ", cf_set.original_label)
print("Counterfactual label:", cf_set.best.label)
print(results.to_dataframe())
```

You can also enforce domain constraints and protect immutable features:

```python
results = explainer.generate_counterfactuals(
    instances_to_explain=X.iloc[:5],
    reference_data=X,
    immutable_features=["age", "gender"],
    relation_constraints=[
        Feature("years_employed") <= Feature("age") - 18,
    ],
)
```

---
## 🆕 What's New in v0.2.0

* **TabularCONFETTI** — counterfactual explanations for tabular data with categorical feature support, immutable features, and Gower distance
* **Relation constraints DSL** — composable inter-feature constraints (`<=`, `<`, `Equal`, `And`, `Or`, `Count`)
* **Rust-accelerated backend** — distances (DTW, Soft-DTW, GAK, Manhattan), NSGA-III components, and constraint evaluation via PyO3
* **Custom NSGA-III** — zero-dependency genetic algorithm (pymoo removed)
* **Built-in distance metrics** — pure-numpy DTW, Soft-DTW, GAK, CTW, Gower, Manhattan (tslearn removed)
* **PyTorch adapter** — use PyTorch models alongside Keras and scikit-learn

---
## 📚 Documentation

The full documentation, including usage guides, API reference, and examples, is available at:

👉 **https://confetti-ts.readthedocs.io/en/latest/**

---
## 📄 License

CONFETTI is released under the [MIT License](LICENSE). 

---
## 📝 Citing CONFETTI
If you use CONFETTI in your research, please consider citing the following paper:

```
@inproceedings{cetina2026counterfactual,
  title={Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification},
  author={Cetina, Alan Gabriel Paredes and Benguessoum, Kaouther and Lourenco, Raoni and Kubler, Sylvain},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={21},
  pages={17393--17400},
  year={2026}
}
```

To **replicate the experiments described in the paper**, use the **`paper` branch** of this
repository. It contains the experiment scripts, model configurations, and dataset handling
used in the publication.
