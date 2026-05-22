from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError
from confetti.tabular._encoder import FeatureEncoder


class TestFeatureEncoderFit:

    def test_fit_records_columns(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        assert enc.columns == ["age", "income"]

    def test_fit_records_dtypes(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        assert enc.dtypes["age"] == numeric_df["age"].dtype

    def test_fit_categorical_mappings(self, mixed_df):
        enc = FeatureEncoder().fit(mixed_df)
        assert "city" in enc.categorical_mappings
        assert "age" not in enc.categorical_mappings
        assert enc.categorical_mappings["city"]["value_to_int"]["LA"] == 0
        assert enc.categorical_mappings["city"]["value_to_int"]["NYC"] == 1

    def test_fit_returns_self(self, numeric_df):
        enc = FeatureEncoder()
        result = enc.fit(numeric_df)
        assert result is enc

    def test_fit_non_dataframe_raises(self):
        with pytest.raises(CONFETTIDataTypeError, match="pd.DataFrame"):
            FeatureEncoder().fit(np.array([1, 2, 3]))

    def test_fit_empty_dataframe_raises(self):
        with pytest.raises(CONFETTIConfigurationError, match="empty"):
            FeatureEncoder().fit(pd.DataFrame())


class TestFeatureEncoderTransform:

    def test_numeric_roundtrip(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        arr = enc.transform(numeric_df)
        assert arr.dtype == np.float64
        assert arr.shape == (len(numeric_df), 2)
        np.testing.assert_array_equal(arr[:, 0], numeric_df["age"].values)

    def test_categorical_encoding(self, mixed_df):
        enc = FeatureEncoder().fit(mixed_df)
        arr = enc.transform(mixed_df)
        assert arr[0, 1] == 1.0  # NYC → 1
        assert arr[1, 1] == 0.0  # LA → 0

    def test_unfitted_raises(self, numeric_df):
        with pytest.raises(CONFETTIConfigurationError, match="before fit"):
            FeatureEncoder().transform(numeric_df)

    def test_column_mismatch_raises(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        wrong_df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(CONFETTIConfigurationError, match="Column mismatch"):
            enc.transform(wrong_df)


class TestFeatureEncoderInverseTransform:

    def test_numeric_roundtrip(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        arr = enc.transform(numeric_df)
        recovered = enc.inverse_transform(arr)
        pd.testing.assert_frame_equal(recovered, numeric_df, check_dtype=False)

    def test_mixed_roundtrip(self, mixed_df):
        enc = FeatureEncoder().fit(mixed_df)
        arr = enc.transform(mixed_df)
        recovered = enc.inverse_transform(arr)
        assert list(recovered.columns) == ["age", "city", "income"]
        assert list(recovered["city"]) == ["NYC", "LA", "NYC", "LA"]

    def test_single_row_roundtrip(self, mixed_df):
        enc = FeatureEncoder().fit(mixed_df)
        single = mixed_df.iloc[0:1]
        arr = enc.transform(single)
        recovered = enc.inverse_transform(arr)
        assert recovered.shape == (1, 3)
        assert recovered["city"].iloc[0] == "NYC"

    def test_1d_input_auto_reshape(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        arr = enc.transform(numeric_df)
        row = arr[0]
        recovered = enc.inverse_transform(row)
        assert recovered.shape == (1, 2)

    def test_wrong_column_count_raises(self, numeric_df):
        enc = FeatureEncoder().fit(numeric_df)
        with pytest.raises(CONFETTIConfigurationError, match="columns"):
            enc.inverse_transform(np.array([[1, 2, 3]]))

    def test_unfitted_raises(self):
        with pytest.raises(CONFETTIConfigurationError, match="before fit"):
            FeatureEncoder().inverse_transform(np.array([[1, 2]]))


class TestFeatureEncoderBooleanColumns:

    def test_boolean_column_roundtrip(self):
        df = pd.DataFrame({
            "active": [True, False, True],
            "score": [10.0, 20.0, 30.0],
        })
        enc = FeatureEncoder().fit(df)
        arr = enc.transform(df)
        assert arr.shape == (3, 2)
        recovered = enc.inverse_transform(arr)
        assert list(recovered.columns) == ["active", "score"]
