Usage
=====

This page describes how to use CONFETTI to generate counterfactual
explanations for multivariate time series.

Overview
--------

CONFETTI generates counterfactual explanations for multivariate time
series using three main stages:

1. **Nearest Unlike Neighbour (NUN) search**
   The explainer identifies, within the reference dataset, the instance
   with a *different* predicted class label and a sufficiently high
   predicted probability (controlled by ``theta``).
   This neighbour guides the perturbation.

2. **Naive subsequence construction (optional)**
   If feature-importance weights (e.g. CAM) are provided, CONFETTI
   extracts the most influential temporal region of the NUN and uses it
   to create a first counterfactual candidate via subsequence
   replacement.

3. **Multi-objective optimization**
   Using NSGA-III, CONFETTI optimizes the perturbation to balance:
   - increasing confidence in the target class
   - minimizing the number of perturbed time steps
   - minimizing proximity to the original instance

The result is a set of counterfactual candidates per input instance,
along with the best-scoring one according to the chosen objective
weight ``alpha``.

Basic Workflow
--------------

To generate counterfactual explanations, you need:

- a **path to a trained model** (Keras ``.keras`` or joblib ``.joblib``)
- one or more **instances to explain** shaped
  ``(n_instances, time_steps, n_features)``
- a **reference dataset** in the same format
  (mandatory for NUN search)
- optional **feature weights** (e.g. CAM values)

.. code-block:: python

    from confetti import CONFETTI
    from confetti.utils import load_multivariate_ts_from_csv
    from confetti.attribution import cam     # optional
    import keras

    # Load model
    model_path = "path/to/model.keras"
    model = keras.models.load_model(model_path)

    # Load datasets
    X_train, y_train = load_multivariate_ts_from_csv("train.csv")
    X_test,  y_test  = load_multivariate_ts_from_csv("test.csv")

    # Prepare instance to explain
    instance = X_test[0:1]

    # Optional: feature-importance weights for the reference data
    reference_weights = cam(model, X_train)

    # Initialize explainer
    explainer = CONFETTI(model_path=model_path)

    # Generate counterfactuals
    results = explainer.generate_counterfactuals(
        instances_to_explain=instance,
        reference_data=X_train,
        reference_weights=reference_weights,   # or None
        alpha=0.5,
        theta=0.51,
    )

Interpreting the Output
-----------------------

``generate_counterfactuals`` returns a ``CounterfactualResults`` object.
This behaves like a list of ``CounterfactualSet`` objects, one for each
instance you provided.

A ``CounterfactualSet`` contains:

- ``original_instance``
- ``original_label``
- ``nearest_unlike_neighbour``
- ``best`` → a ``Counterfactual`` object
- ``all_counterfactuals`` → list of all candidates generated
- ``feature_importance`` → 1D array of the NUN CAM weights (if provided)

Example:

.. code-block:: python

    cf_set = results[0]

    original  = cf_set.original_instance
    nun       = cf_set.nearest_unlike_neighbour
    best_cf   = cf_set.best.counterfactual
    best_label = cf_set.best.label
    all_cfs   = cf_set.all_counterfactuals
    weights   = cf_set.feature_importance

Exporting Counterfactuals
-------------------------

You can export results to a pandas DataFrame or save them as CSV files.

.. code-block:: python

    df = results.to_dataframe()
    results.to_csv("all_counterfactuals.csv")

Visualizing Explanations
------------------------

CONFETTI includes simple visualization utilities for inspecting
counterfactual explanations.

.. code-block:: python

    from confetti.visualizations import plot_counterfactual

    plot_counterfactual(
        original=cf_set.original_instance,
        counterfactual=cf_set.best.counterfactual,
        cam_weights=cf_set.feature_importance,
        cam_mode="heatmap",
    )

The resulting plot highlights:

- the original time series
- the modified subsequence used for the counterfactual
- (optionally) the CAM heatmap used to guide perturbation

This helps interpret both *what* changed and *why* the model's decision
was flipped.

Further Reading
---------------

For a runnable end-to-end example, see the :doc:`example` page.
