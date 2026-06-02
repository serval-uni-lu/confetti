Tabular Example
===============

This example walks through the full ``TabularCONFETTI`` pipeline for
generating counterfactual explanations on tabular data, including:

- preparing a mixed-type dataset
- training a classifier
- generating counterfactual candidates
- using immutable features and relation constraints
- choosing a proximity metric for mixed-type data


------------------------------------------------------------
1. Prepare a Dataset
------------------------------------------------------------

We create a small synthetic dataset with numerical and categorical
features.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n = 300

    df = pd.DataFrame({
        "age": rng.integers(18, 65, size=n),
        "income": rng.normal(50_000, 15_000, size=n).round(2),
        "education": rng.choice(["high_school", "bachelor", "master", "phd"], size=n),
        "years_employed": rng.integers(0, 30, size=n),
    })

    # Target: approved if income > 40k and age > 25
    df["approved"] = ((df["income"] > 40_000) & (df["age"] > 25)).astype(int)

    X = df.drop(columns=["approved"])
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train:", X_train.shape)
    print("Test: ", X_test.shape)

.. code-block:: text

    Train: (240, 4)
    Test:  (60, 4)


------------------------------------------------------------
2. Train a Classifier
------------------------------------------------------------

We use a ``RandomForestClassifier`` from scikit-learn.  Any classifier
with ``predict_proba`` works.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier

    # Encode categoricals for the model
    X_train_enc = pd.get_dummies(X_train)
    X_test_enc = pd.get_dummies(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_enc, y_train)

    accuracy = model.score(X_test_enc, y_test)
    print(f"Test accuracy: {accuracy:.3f}")


------------------------------------------------------------
3. Generate Counterfactual Explanations
------------------------------------------------------------

Initialize ``TabularCONFETTI`` and generate counterfactuals.  Since the
model was trained on one-hot encoded data, we pass a ``preprocessor``
that applies the same encoding before prediction.

.. code-block:: python

    from confetti import TabularCONFETTI

    explainer = TabularCONFETTI(
        model,
        preprocessor=lambda X: pd.get_dummies(X),
    )

    results = explainer.generate_counterfactuals(
        instances_to_explain=X_test.iloc[:3],
        reference_data=X_train,
        alpha=0.5,
        theta=0.51,
    )


------------------------------------------------------------
4. Inspect Results
------------------------------------------------------------

The output is a ``CounterfactualResults`` object, the same structure
used by the time-series explainer.

.. code-block:: python

    for i, cf_set in enumerate(results):
        print(f"\n--- Instance {i} ---")
        print(f"Original label:      {cf_set.original_label}")
        print(f"Counterfactual label: {cf_set.best.label}")
        print(f"Candidates generated: {len(cf_set.all_counterfactuals)}")

    # Export all counterfactuals as a DataFrame
    df_results = results.to_dataframe()
    print(df_results)


------------------------------------------------------------
5. Immutable Features
------------------------------------------------------------

Prevent specific features from being changed in counterfactuals.  This
is useful when certain features are actionable but others are not (e.g.
age, gender).

.. code-block:: python

    results = explainer.generate_counterfactuals(
        instances_to_explain=X_test.iloc[:3],
        reference_data=X_train,
        immutable_features=["age"],
    )

    # Verify: age is unchanged in every counterfactual
    for cf_set in results:
        original_age = cf_set.original_instance["age"].values[0]
        cf_age = cf_set.best.counterfactual["age"].values[0]
        assert original_age == cf_age


------------------------------------------------------------
6. Relation Constraints
------------------------------------------------------------

Enforce domain rules on counterfactual explanations.  For example,
years employed should not exceed the person's age minus 18:

.. code-block:: python

    from confetti import Feature

    constraints = [
        Feature("years_employed") <= Feature("age") - 18,
    ]

    results = explainer.generate_counterfactuals(
        instances_to_explain=X_test.iloc[:3],
        reference_data=X_train,
        relation_constraints=constraints,
    )

See the :doc:`constraints` page for the full constraint DSL reference.


------------------------------------------------------------
7. Gower Distance for Mixed-Type Data
------------------------------------------------------------

When your data contains both numerical and categorical features, use
Gower distance as the proximity metric.

.. code-block:: python

    results = explainer.generate_counterfactuals(
        instances_to_explain=X_test.iloc[:3],
        reference_data=X_train,
        optimize_proximity=True,
        proximity_distance="gower",
    )

Gower distance computes binary match/mismatch for categorical columns
and range-normalized absolute difference for numerical columns.  When
the input is a ``DataFrame``, categorical columns are auto-detected.
You can also specify them explicitly:

.. code-block:: python

    results = explainer.generate_counterfactuals(
        instances_to_explain=X_test.iloc[:3],
        reference_data=X_train,
        optimize_proximity=True,
        proximity_distance="gower",
        categorical_features=[2],  # index of "education" column
    )
