from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError
from confetti.structs import Counterfactual, CounterfactualResults
from confetti.tabular import TabularCONFETTI


class HighLowClassifier:
    """Deterministic classifier: class 1 when mean feature value >= 50, else class 0.

    Produces a smooth probability curve so the GA can optimise.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = 1 / (1 + np.exp(-(np.mean(X, axis=1) - 50) / 10))
        return np.column_stack([1 - score, score])


@pytest.fixture
def high_low_clf():
    return HighLowClassifier()


@pytest.fixture
def tabular_df():
    return pd.DataFrame({
        "age": [20, 25, 30, 60, 70, 80],
        "income": [10, 15, 20, 80, 85, 90],
    })


@pytest.fixture
def tabular_np():
    return np.array([
        [20, 10],
        [25, 15],
        [30, 20],
        [60, 80],
        [70, 85],
        [80, 90],
    ], dtype=np.float64)


class TestInitValidation:

    def test_model_without_predict_raises(self):
        with pytest.raises(CONFETTIConfigurationError, match="predict"):
            TabularCONFETTI(model=object())

    def test_feature_names_not_list_raises(self, high_low_clf):
        with pytest.raises(CONFETTIConfigurationError, match="list of strings"):
            TabularCONFETTI(model=high_low_clf, feature_names="age")

    def test_feature_names_duplicate_raises(self, high_low_clf):
        with pytest.raises(CONFETTIConfigurationError, match="duplicates"):
            TabularCONFETTI(model=high_low_clf, feature_names=["a", "a"])

    def test_valid_init(self, high_low_clf):
        explainer = TabularCONFETTI(model=high_low_clf, feature_names=["age", "income"])
        assert explainer.model is high_low_clf
        assert explainer.feature_names == ["age", "income"]


class TestCounterfactualParamsValidation:

    def test_instances_wrong_type_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIDataTypeError, match="instances_to_explain"):
            explainer.generate_counterfactuals(
                instances_to_explain=[1, 2],
                reference_data=tabular_np,
            )

    def test_reference_wrong_type_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIDataTypeError, match="reference_data"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np,
                reference_data="bad",
            )

    def test_3d_instances_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="2-D"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np.reshape(2, 3, 2),
                reference_data=tabular_np,
            )

    def test_feature_count_mismatch_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="Feature count"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:, :1],
                reference_data=tabular_np,
            )

    def test_column_name_mismatch_raises(self, high_low_clf):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"x": [1], "y": [2]})
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="Column names"):
            explainer.generate_counterfactuals(
                instances_to_explain=df1,
                reference_data=df2,
            )

    def test_alpha_out_of_range_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="alpha"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                alpha=1.5,
            )

    def test_theta_out_of_range_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="theta"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                theta=0.0,
            )

    def test_unsupported_metric_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="Unsupported"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                proximity_distance="dtw",
            )

    def test_fewer_than_two_objectives_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="two objectives"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                optimize_confidence=True,
                optimize_sparsity=False,
                optimize_proximity=False,
            )

    def test_bad_population_size_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="population_size"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                population_size=-1,
            )

    def test_bad_processes_raises(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        with pytest.raises(CONFETTIConfigurationError, match="processes"):
            explainer.generate_counterfactuals(
                instances_to_explain=tabular_np[:1],
                reference_data=tabular_np,
                processes=0,
            )


class TestEndToEndNumpy:

    def test_basic_generation(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf)
        results = explainer.generate_counterfactuals(
            instances_to_explain=tabular_np[:1],
            reference_data=tabular_np,
            population_size=20,
            maximum_number_of_generations=10,
        )
        assert results is not None
        assert isinstance(results, CounterfactualResults)
        assert len(results) >= 1
        assert results[0].best is not None

    def test_feature_names_in_output(self, high_low_clf, tabular_np):
        explainer = TabularCONFETTI(model=high_low_clf, feature_names=["age", "income"])
        results = explainer.generate_counterfactuals(
            instances_to_explain=tabular_np[:1],
            reference_data=tabular_np,
            population_size=20,
            maximum_number_of_generations=10,
        )
        assert results is not None
        best_cf = results[0].best.counterfactual
        assert isinstance(best_cf, pd.DataFrame)
        assert list(best_cf.columns) == ["age", "income"]


class TestEndToEndDataFrame:

    def test_dataframe_input_output(self, high_low_clf, tabular_df):
        explainer = TabularCONFETTI(model=high_low_clf)
        results = explainer.generate_counterfactuals(
            instances_to_explain=tabular_df.iloc[:1],
            reference_data=tabular_df,
            population_size=20,
            maximum_number_of_generations=10,
        )
        assert results is not None
        best_cf = results[0].best.counterfactual
        assert isinstance(best_cf, pd.DataFrame)
        assert list(best_cf.columns) == ["age", "income"]


class TestCategoricalFeatures:

    def test_categorical_values_in_output(self, high_low_clf):
        df = pd.DataFrame({
            "score": [10, 15, 20, 80, 85, 90],
            "color": ["red", "red", "red", "blue", "blue", "blue"],
        })
        explainer = TabularCONFETTI(model=high_low_clf)
        results = explainer.generate_counterfactuals(
            instances_to_explain=df.iloc[:1],
            reference_data=df,
            population_size=20,
            maximum_number_of_generations=10,
        )
        if results is not None and results[0].best is not None:
            best_cf = results[0].best.counterfactual
            assert isinstance(best_cf, pd.DataFrame)
            assert best_cf["color"].iloc[0] in ("red", "blue")


class TestPredictOnlyModel:

    def test_predict_only_model_works(self, tabular_np):
        from tests.tabular.conftest import MockPredictOnlyClassifier
        clf = MockPredictOnlyClassifier()
        explainer = TabularCONFETTI(model=clf)
        results = explainer.generate_counterfactuals(
            instances_to_explain=tabular_np[:1],
            reference_data=tabular_np,
            population_size=20,
            maximum_number_of_generations=10,
        )
        assert results is None or isinstance(results, CounterfactualResults)


class TestCounterfactualEquality:

    def test_dataframe_counterfactual_equality(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [1], "b": [2]})
        cf1 = Counterfactual(counterfactual=df1, label=0)
        cf2 = Counterfactual(counterfactual=df2, label=0)
        assert cf1 == cf2

    def test_dataframe_counterfactual_inequality(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "b": [4]})
        cf1 = Counterfactual(counterfactual=df1, label=0)
        cf2 = Counterfactual(counterfactual=df2, label=0)
        assert cf1 != cf2

    def test_mixed_type_inequality(self):
        arr = np.array([1, 2])
        df = pd.DataFrame({"a": [1], "b": [2]})
        cf1 = Counterfactual(counterfactual=arr, label=0)
        cf2 = Counterfactual(counterfactual=df, label=0)
        assert cf1 != cf2

    def test_ndarray_equality_still_works(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        cf1 = Counterfactual(counterfactual=arr1, label=1)
        cf2 = Counterfactual(counterfactual=arr2, label=1)
        assert cf1 == cf2
