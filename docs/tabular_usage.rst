Tabular Usage
=============

.. note::
   This page covers **tabular** counterfactual generation.
   For multivariate time series, see :doc:`usage`.

This page describes how to use ``TabularCONFETTI`` to generate
counterfactual explanations for tabular (non-time-series) classifiers.

Overview
--------

``TabularCONFETTI`` generates counterfactual explanations for tabular
data using the same NSGA-III multi-objective engine as the time-series
explainer, adapted for feature-level perturbations:

- **Decision variables** are a binary mask over features (not time
  windows).  For each feature, the GA decides whether to keep the
  original value or adopt the value from the nearest unlike neighbour
  (NUN).
- **Input** is a 2D ``DataFrame`` or ``ndarray`` of shape
  ``(n_instances, n_features)`` rather than a 3D time-series array.
- **A single GA run** replaces the time-series window binary search.
- **Categorical features** are automatically encoded and decoded when
  the input is a ``DataFrame``.

The optimization objectives are the same as for time series:
increasing confidence in the target class, minimizing the number of
changed features (sparsity), and optionally minimizing proximity to
the original instance.


Basic Workflow
--------------

To generate tabular counterfactuals you need:

- a **fitted classifier** with ``predict_proba()`` or ``predict()``
  returning class probabilities
- one or more **instances to explain** shaped
  ``(n_instances, n_features)``
- a **reference dataset** in the same format

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from confetti import TabularCONFETTI

    # Prepare data
    df_train = pd.read_csv("train.csv")
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    # Train a classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Select instances to explain
    instances = X_train.iloc[:5]

    # Initialize explainer
    explainer = TabularCONFETTI(model)

    # Generate counterfactuals
    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        alpha=0.5,
        theta=0.51,
    )

    # Inspect the best counterfactual for the first instance
    cf_set = results[0]
    print("Original label:     ", cf_set.original_label)
    print("Counterfactual label:", cf_set.best.label)


Categorical Features
--------------------

When the input is a ``DataFrame`` with non-numeric columns (e.g.
``object`` or ``category`` dtype), ``TabularCONFETTI`` automatically
encodes them to ordinal integers for the GA and decodes back to the
original labels when returning results.

.. code-block:: python

    df = pd.DataFrame({
        "age": [25, 40, 35],
        "job": ["engineer", "teacher", "doctor"],
        "income": [50000, 45000, 70000],
        "target": [0, 1, 0],
    })

    X = df.drop(columns=["target"])
    y = df["target"]

    model.fit(X.select_dtypes(include="number"), y)

    explainer = TabularCONFETTI(model)
    results = explainer.generate_counterfactuals(
        instances_to_explain=X.iloc[:1],
        reference_data=X,
    )

    # Counterfactual DataFrames show original labels, not ordinal codes
    print(results.to_dataframe())

No extra configuration is needed — detection is automatic.


Immutable Features
------------------

Use the ``immutable_features`` parameter to prevent specific features
from being modified in counterfactual explanations.  Accepts column
names (``list[str]``) when the input is a ``DataFrame`` or
``feature_names`` were provided, or column indices (``list[int]``)
for any input type.

.. code-block:: python

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        immutable_features=["age", "gender"],
    )

Immutable features are guaranteed to retain their original values and
are excluded from the sparsity objective.


Relation Constraints
--------------------

Use the ``relation_constraints`` parameter to enforce domain rules on
generated counterfactuals.  Constraints are built from ``Feature``
references, constants, and comparison operators.

.. code-block:: python

    from confetti import Feature

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        relation_constraints=[
            Feature("years_employed") <= Feature("age") - 18,
            Feature("debt") <= Feature("income") * 0.4,
        ],
    )

The GA treats constraint violations as inequality constraints, steering
the search toward counterfactuals that satisfy all rules.  Constraints
can be combined with ``&`` (all must hold) and ``|`` (at least one must
hold), and ``Equal`` constraints with a single ``Feature`` on the left
side are repaired in-place for faster convergence.

Relation constraints compose freely with ``immutable_features`` — you
can lock a feature and still reference it in constraints:

.. code-block:: python

    from confetti import Feature, Equal

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        immutable_features=["age"],
        relation_constraints=[
            Feature("years_employed") <= Feature("age") - 18,
            Equal(Feature("total_income"), Feature("salary") + Feature("bonus")),
        ],
    )

See :doc:`constraints` for the full constraint DSL reference, including
``SafeDivision``, ``Log``, ``ManySum``, ``Count``, and more.


Proximity Metrics
-----------------

By default, when ``optimize_proximity=True`` the proximity objective
uses Euclidean distance.  Three metrics are available:

- ``"euclidean"`` — standard L2 distance (default)
- ``"manhattan"`` — L1 distance
- ``"gower"`` — Gower distance for mixed-type data (categorical +
  numerical)

Gower distance is the recommended choice when your data contains
categorical features.  It computes binary match/mismatch for
categorical columns and range-normalized absolute difference for
numerical columns.

.. code-block:: python

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        optimize_proximity=True,
        proximity_distance="gower",
        categorical_features=[0, 3, 5],  # or auto-detected from DataFrame
    )

When ``proximity_distance="gower"`` and ``categorical_features`` is
not provided, categorical columns are auto-detected from the
``DataFrame`` schema.  When all features are numerical, Gower reduces
to range-normalized Manhattan distance.


Using a Preprocessor
--------------------

When your model expects a different encoding than the raw feature space
(e.g. one-hot encoding, standard scaling), provide a ``preprocessor``
callback.  It is applied before every call to the model, while the GA
search, NUN lookup, and objectives all operate on the original feature
space.

.. code-block:: python

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)

    explainer = TabularCONFETTI(
        model,
        preprocessor=lambda X: scaler.transform(X),
    )


Interpreting the Output
-----------------------

``generate_counterfactuals`` returns the same ``CounterfactualResults``
structure as the time-series explainer.  See :ref:`the output guide
<interpreting-the-output>` in the time-series usage page for full
details.

When the input was a ``DataFrame`` with categorical columns, the
counterfactuals in the result are also ``DataFrame`` objects with the
original category labels restored.


Further Reading
---------------

- :doc:`tabular_example` — end-to-end tabular example
- :doc:`constraints` — inter-feature relational constraints
- :doc:`api/confetti.tabular` — API reference
