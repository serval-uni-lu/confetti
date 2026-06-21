"""Tests for the counterfactual dataclasses (structs/)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confetti.errors import CONFETTIDataTypeError, CONFETTIError
from confetti.structs import Counterfactual, CounterfactualSet, CounterfactualResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cf_array():
    return np.arange(6, dtype=float).reshape(3, 2)


@pytest.fixture
def cf_array_alt():
    return np.ones((3, 2), dtype=float) * 9.0


@pytest.fixture
def original_instance():
    return np.zeros((3, 2), dtype=float)


@pytest.fixture
def nun_instance():
    return np.full((3, 2), 5.0, dtype=float)


@pytest.fixture
def counterfactual_a(cf_array):
    return Counterfactual(counterfactual=cf_array, label=1)


@pytest.fixture
def counterfactual_b(cf_array_alt):
    return Counterfactual(counterfactual=cf_array_alt, label=0)


@pytest.fixture
def valid_set(original_instance, nun_instance, counterfactual_a, counterfactual_b):
    return CounterfactualSet(
        original_instance=original_instance,
        original_label=0,
        nearest_unlike_neighbour=nun_instance,
        best_solution=counterfactual_a,
        all_counterfactuals=[counterfactual_a, counterfactual_b],
    )


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------


class TestCounterfactual:
    def test_equality_same_content(self, cf_array):
        a = Counterfactual(counterfactual=cf_array.copy(), label=1)
        b = Counterfactual(counterfactual=cf_array.copy(), label=1)
        assert a == b

    def test_inequality_different_label(self, cf_array):
        a = Counterfactual(counterfactual=cf_array, label=1)
        b = Counterfactual(counterfactual=cf_array.copy(), label=0)
        assert a != b

    def test_inequality_different_array(self, cf_array, cf_array_alt):
        a = Counterfactual(counterfactual=cf_array, label=1)
        b = Counterfactual(counterfactual=cf_array_alt, label=1)
        assert a != b

    def test_equality_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        a = Counterfactual(counterfactual=df.copy(), label="x")
        b = Counterfactual(counterfactual=df.copy(), label="x")
        assert a == b

    def test_inequality_mixed_types(self, cf_array):
        a = Counterfactual(counterfactual=cf_array, label=1)
        b = Counterfactual(
            counterfactual=pd.DataFrame(cf_array), label=1
        )
        assert a != b

    def test_not_implemented_for_non_counterfactual(self, cf_array):
        a = Counterfactual(counterfactual=cf_array, label=1)
        assert a.__eq__("not a counterfactual") is NotImplemented


# ---------------------------------------------------------------------------
# CounterfactualSet — type validation
# ---------------------------------------------------------------------------


class TestCounterfactualSetValidation:
    def test_valid_construction(self, valid_set, original_instance):
        assert np.array_equal(valid_set.original_instance, original_instance)
        assert valid_set.original_label == 0

    def test_accepts_numpy_label_types(self, original_instance, nun_instance, counterfactual_a):
        for label in [np.int64(1), np.float64(0.5)]:
            cs = CounterfactualSet(
                original_instance=original_instance,
                original_label=label,
                nearest_unlike_neighbour=nun_instance,
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
            )
            assert cs.original_label == label

    def test_rejects_bad_original_instance(self, nun_instance, counterfactual_a):
        with pytest.raises(CONFETTIDataTypeError, match="original_instance"):
            CounterfactualSet(
                original_instance="not_an_array",
                original_label=0,
                nearest_unlike_neighbour=nun_instance,
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
            )

    def test_rejects_bad_original_label(self, original_instance, nun_instance, counterfactual_a):
        with pytest.raises(CONFETTIDataTypeError, match="original_label"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=[0, 1],
                nearest_unlike_neighbour=nun_instance,
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
            )

    def test_rejects_bad_nun(self, original_instance, counterfactual_a):
        with pytest.raises(CONFETTIDataTypeError, match="nearest_unlike_neighbour"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=0,
                nearest_unlike_neighbour="bad",
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
            )

    def test_rejects_bad_best_solution(self, original_instance, nun_instance):
        with pytest.raises(CONFETTIDataTypeError, match="best_solution"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=0,
                nearest_unlike_neighbour=nun_instance,
                best_solution="not_a_cf",
                all_counterfactuals=[],
            )

    def test_rejects_bad_all_counterfactuals(self, original_instance, nun_instance):
        with pytest.raises(CONFETTIDataTypeError, match="all_counterfactuals"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=0,
                nearest_unlike_neighbour=nun_instance,
                best_solution=None,
                all_counterfactuals=["not_a_cf"],
            )

    def test_rejects_non_array_feature_importance(self, original_instance, nun_instance, counterfactual_a):
        with pytest.raises(CONFETTIDataTypeError, match="feature_importance"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=0,
                nearest_unlike_neighbour=nun_instance,
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
                feature_importance=[1, 2, 3],
            )

    def test_rejects_2d_feature_importance(self, original_instance, nun_instance, counterfactual_a):
        with pytest.raises(CONFETTIDataTypeError, match="feature_importance"):
            CounterfactualSet(
                original_instance=original_instance,
                original_label=0,
                nearest_unlike_neighbour=nun_instance,
                best_solution=counterfactual_a,
                all_counterfactuals=[counterfactual_a],
                feature_importance=np.ones((2, 3)),
            )

    def test_accepts_valid_feature_importance(self, original_instance, nun_instance, counterfactual_a):
        fi = np.array([0.5, 0.3, 0.2])
        cs = CounterfactualSet(
            original_instance=original_instance,
            original_label=0,
            nearest_unlike_neighbour=nun_instance,
            best_solution=counterfactual_a,
            all_counterfactuals=[counterfactual_a],
            feature_importance=fi,
        )
        np.testing.assert_array_equal(cs.feature_importance, fi)


# ---------------------------------------------------------------------------
# CounterfactualSet — best property
# ---------------------------------------------------------------------------


class TestCounterfactualSetBest:
    def test_best_returns_designated_best(self, valid_set, counterfactual_a):
        assert valid_set.best == counterfactual_a

    def test_best_none_when_set_to_none(self, original_instance, nun_instance, counterfactual_a):
        cs = CounterfactualSet(
            original_instance=original_instance,
            original_label=0,
            nearest_unlike_neighbour=nun_instance,
            best_solution=None,
            all_counterfactuals=[counterfactual_a],
        )
        assert cs.best is None


# ---------------------------------------------------------------------------
# CounterfactualSet — to_dataframe
# ---------------------------------------------------------------------------


class TestCounterfactualSetDataFrame:
    def test_to_dataframe_shape(self, valid_set):
        df = valid_set.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_to_dataframe_marks_best(self, valid_set):
        df = valid_set.to_dataframe()
        assert df["is_best"].sum() == 1
        assert df.loc[df["is_best"]].iloc[0]["label"] == valid_set.best.label

    def test_to_dataframe_empty_raises(self, original_instance, nun_instance):
        cs = CounterfactualSet(
            original_instance=original_instance,
            original_label=0,
            nearest_unlike_neighbour=nun_instance,
            best_solution=None,
            all_counterfactuals=[],
        )
        with pytest.raises(CONFETTIError, match="No counterfactuals available"):
            cs.to_dataframe()


# ---------------------------------------------------------------------------
# CounterfactualResults
# ---------------------------------------------------------------------------


class TestCounterfactualResults:
    def test_len_empty(self):
        cr = CounterfactualResults()
        assert len(cr) == 0

    def test_len_with_sets(self, valid_set):
        cr = CounterfactualResults(counterfactual_sets=[valid_set, valid_set])
        assert len(cr) == 2

    def test_getitem(self, valid_set):
        cr = CounterfactualResults(counterfactual_sets=[valid_set])
        assert cr[0] is valid_set

    def test_getitem_none_raises(self):
        cr = CounterfactualResults()
        with pytest.raises(CONFETTIError, match="No counterfactual sets"):
            cr[0]

    def test_to_dataframe(self, valid_set):
        cr = CounterfactualResults(counterfactual_sets=[valid_set, valid_set])
        df = cr.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_to_dataframe_empty_raises(self):
        cr = CounterfactualResults()
        with pytest.raises(CONFETTIError, match="No counterfactual sets"):
            cr.to_dataframe()

    def test_iteration_via_index(self, valid_set):
        cr = CounterfactualResults(counterfactual_sets=[valid_set])
        for i in range(len(cr)):
            assert isinstance(cr[i], CounterfactualSet)
