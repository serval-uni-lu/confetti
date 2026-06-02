Relation Constraints
====================

Relation constraints let you encode domain knowledge as rules that
counterfactual explanations must satisfy.  For example, you can require
that a person's years of employment never exceed their age minus 18, or
that two features maintain a specific ratio.

Constraints are passed to ``generate_counterfactuals()`` via the
``relation_constraints`` parameter.  The GA treats constraint violations
as inequality constraints, steering the search toward feasible
counterfactuals.

.. code-block:: python

    from confetti import Feature, TabularCONFETTI

    constraints = [
        Feature("years_employed") <= Feature("age") - 18,
        Feature("debt") <= Feature("income") * 0.4,
    ]

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=reference,
        relation_constraints=constraints,
    )


Value Expressions
-----------------

Value expressions represent quantities in the constraint language.
They can reference feature columns, literal constants, or arithmetic
combinations of other values.

``Feature(feature_id)``
    A reference to a feature column by name (``str``) or zero-based
    index (``int``).

    .. code-block:: python

        from confetti import Feature
        age = Feature("age")
        col_0 = Feature(0)

``Constant(value)``
    A literal numeric constant.

    .. code-block:: python

        from confetti import Constant
        threshold = Constant(18)

    .. note::
       Plain Python numbers (``int`` and ``float``) are automatically
       coerced to ``Constant`` when used in arithmetic with value
       expressions, so ``Feature("age") - 18`` works without wrapping
       ``18`` in ``Constant()``.

**Arithmetic operators** are supported between value expressions:

.. code-block:: python

    ratio = Feature("income") / Feature("household_size")
    total = Feature("salary") + Feature("bonus") * 2
    squared = Feature("x") ** 2

All standard operators work: ``+``, ``-``, ``*``, ``/``, ``**``,
``%``.

**Additional value nodes**:

``SafeDivision(dividend, divisor, fill_value)``
    Division that falls back to ``fill_value`` when the divisor is
    zero, avoiding division errors.

    .. code-block:: python

        from confetti import SafeDivision, Feature, Constant
        ratio = SafeDivision(Feature("debt"), Feature("income"), Constant(0))

``ManySum(operands)``
    N-ary sum of two or more value expressions.

    .. code-block:: python

        from confetti import ManySum, Feature
        total = ManySum([Feature("a"), Feature("b"), Feature("c")])

``Log(operand, safe_value=None)``
    Natural logarithm.  When ``safe_value`` is provided, it is used as
    a fallback for non-positive inputs.

    .. code-block:: python

        from confetti import Log, Feature, Constant
        log_income = Log(Feature("income"), safe_value=Constant(0))


Constraint Predicates
---------------------

Constraints express relations between value expressions.  Each
constraint computes a soft violation magnitude: zero means the
constraint is satisfied, positive values indicate how far it is from
being met.

**Comparison operators** on value expressions produce constraints
directly:

.. code-block:: python

    c1 = Feature("age") <= Feature("retirement_age")   # LessEqual
    c2 = Feature("debt") < Feature("income")            # Less

``Equal(left, right, tolerance=None)``
    Equality constraint.  When ``tolerance`` is provided, the
    constraint is satisfied when ``|left - right| <= tolerance``.

    .. note::
       Python's ``==`` operator is not overloaded (it would break
       hashing).  Use the ``Equal()`` constructor directly.

    .. code-block:: python

        from confetti import Equal, Feature, Constant
        c = Equal(Feature("total"), Feature("a") + Feature("b"))
        c_approx = Equal(Feature("x"), Constant(10), tolerance=Constant(0.5))


Combinators
-----------

Constraints can be combined using logical operators:

``&`` (And)
    All child constraints must hold.  Violation is the sum of
    individual violations.

    .. code-block:: python

        both = (Feature("a") <= Feature("b")) & (Feature("c") < Feature("d"))

``|`` (Or)
    At least one child constraint must hold.  Violation is the minimum
    of individual violations.

    .. code-block:: python

        either = (Feature("x") <= Constant(10)) | (Feature("y") <= Constant(20))

``Count(operands, inverse=False)``
    A value node that counts the number of violated (or satisfied, if
    ``inverse=True``) constraints per sample.  Since ``Count`` is a
    value expression, it can be used inside other constraints.

    .. code-block:: python

        from confetti import Count, Feature, Constant

        rules = [
            Feature("a") <= Constant(10),
            Feature("b") <= Constant(20),
            Feature("c") <= Constant(30),
        ]

        # Require at most 1 violated constraint
        c = Count(rules) <= Constant(1)


Equality Repair
---------------

``Equal`` constraints whose left operand is a single ``Feature`` are
repaired in-place on counterfactuals before objective computation.
This means the GA will directly set the feature value to satisfy the
equality, rather than relying solely on evolutionary pressure.

End-to-End Example
------------------

The following example ties together several constraint types in a
realistic loan-approval scenario.  The dataset has features ``age``,
``years_employed``, ``income``, ``debt``, and ``education_years``.

.. code-block:: python

    from confetti import (
        Feature,
        Constant,
        Equal,
        SafeDivision,
        Log,
        TabularCONFETTI,
    )

    # --- Value expressions ------------------------------------------------
    age = Feature("age")
    years_employed = Feature("years_employed")
    income = Feature("income")
    debt = Feature("debt")
    education_years = Feature("education_years")

    # --- Constraints ------------------------------------------------------

    # 1. Years employed cannot exceed age minus 18
    c_employment = years_employed <= age - 18

    # 2. Debt-to-income ratio must stay below 40 %
    #    Use SafeDivision to avoid errors when income is zero.
    dti = SafeDivision(debt, income, Constant(0))
    c_dti = dti <= Constant(0.4)

    # 3. Education years + years employed cannot exceed age minus 5
    c_timeline = education_years + years_employed <= age - 5

    # 4. Log-income must be at least 9.2 (≈ $10k)
    c_min_income = Constant(9.2) <= Log(income, safe_value=Constant(0))

    # 5. Total experience must equal education + employment (equality repair)
    total_experience = Feature("total_experience")
    c_total = Equal(total_experience, education_years + years_employed)

    # --- Combine with & / | ----------------------------------------------

    # All constraints must hold simultaneously
    all_constraints = [c_employment, c_dti, c_timeline, c_min_income, c_total]

    # Alternatively, combine a subset with Or:
    #   "either debt is below 40 % of income OR income is above 60k"
    c_flexible = c_dti | (Constant(60_000) <= income)

    # --- Generate counterfactuals -----------------------------------------

    explainer = TabularCONFETTI(model)

    results = explainer.generate_counterfactuals(
        instances_to_explain=instances,
        reference_data=X_train,
        immutable_features=["age"],          # age cannot change
        relation_constraints=all_constraints, # all rules enforced
    )

    # Inspect the best counterfactual
    cf = results[0].best.counterfactual
    print(cf)

Key points illustrated above:

- **Arithmetic composition**: ``education_years + years_employed``
  builds a ``MathOperation`` that can be used inside constraints or
  wrapped in ``Equal`` for repair.
- **SafeDivision**: avoids division-by-zero in the debt-to-income
  ratio without crashing the GA.
- **Log**: lets you express constraints in log-space (e.g. minimum
  income thresholds).
- **Equal + repair**: ``c_total`` is automatically repaired in-place
  because its left operand is a single ``Feature``.
- **Immutable + constraints**: ``age`` is locked, but still referenced
  in ``c_employment`` and ``c_timeline`` — the GA reads its value
  without modifying it.
- **Or combinator**: ``c_flexible`` shows how to express "at least one
  of these must hold."


API Reference
-------------

See the :doc:`api/confetti.constraints` page for the full class and
method reference.
