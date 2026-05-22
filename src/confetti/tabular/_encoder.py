from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError


class FeatureEncoder:
    """Encode a mixed-type DataFrame to a float ndarray and back.

    Records column names, dtypes, and categorical value mappings on
    :meth:`fit`, then converts between DataFrame and NumPy
    representations via :meth:`transform` and :meth:`inverse_transform`.

    Categorical columns are ordinal-encoded (value → int). Numeric
    columns pass through unchanged. The encoding is designed for
    faithful roundtripping, not for distance semantics.

    Parameters
    ----------
    None

    Attributes
    ----------
    columns : list[str] or None
        Column names learned during :meth:`fit`.
    dtypes : dict[str, np.dtype] or None
        Original dtype per column learned during :meth:`fit`.
    categorical_mappings : dict[str, dict] or None
        Per-column ``{value: int}`` and ``{int: value}`` mappings for
        categorical columns.
    """

    def __init__(self) -> None:
        self.columns: Optional[list[str]] = None
        self.dtypes: Optional[dict[str, np.dtype]] = None
        self.categorical_mappings: Optional[dict[str, dict]] = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the encoder has been fitted."""
        return self.columns is not None

    def fit(self, df: pd.DataFrame) -> FeatureEncoder:
        """Learn column names, dtypes, and categorical mappings.

        Parameters
        ----------
        ``df`` : pd.DataFrame
            Reference DataFrame whose schema will be used for all
            subsequent transforms.

        Returns
        -------
        FeatureEncoder
            self

        Raises
        ------
        CONFETTIDataTypeError
            If *df* is not a ``pd.DataFrame``.
        CONFETTIConfigurationError
            If *df* is empty (zero rows or zero columns).
        """
        if not isinstance(df, pd.DataFrame):
            raise CONFETTIDataTypeError(
                message=f"FeatureEncoder.fit() expects a pd.DataFrame, got {type(df).__name__}.",
                param="df",
                hint="Pass a pandas DataFrame.",
            )
        if df.empty or df.shape[1] == 0:
            raise CONFETTIConfigurationError(
                message="Cannot fit an encoder on an empty DataFrame.",
                param="df",
                hint="Provide a DataFrame with at least one row and one column.",
            )

        self.columns = list(df.columns)
        self.dtypes = {col: df[col].dtype for col in self.columns}
        self.categorical_mappings = {}

        for col in self.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                unique_values = sorted(df[col].dropna().unique(), key=str)
                value_to_int = {val: idx for idx, val in enumerate(unique_values)}
                int_to_value = {idx: val for val, idx in value_to_int.items()}
                self.categorical_mappings[col] = {
                    "value_to_int": value_to_int,
                    "int_to_value": int_to_value,
                }

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convert a DataFrame to a float64 ndarray.

        Categorical columns are replaced with their ordinal encoding.
        Numeric columns are cast to float64 as-is.

        Parameters
        ----------
        ``df`` : pd.DataFrame
            DataFrame with columns matching those seen during
            :meth:`fit`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_rows, n_features)`` with dtype
            ``float64``.

        Raises
        ------
        CONFETTIConfigurationError
            If the encoder has not been fitted or if columns do not
            match.
        """
        columns, dtypes, cat_mappings = self._require_fitted("transform")
        self._check_columns(df, "transform")

        result = np.empty((len(df), len(columns)), dtype=np.float64)

        for i, col in enumerate(columns):
            if col in cat_mappings:
                mapping = cat_mappings[col]["value_to_int"]
                result[:, i] = df[col].map(mapping).to_numpy(dtype=np.float64)
            else:
                result[:, i] = df[col].to_numpy(dtype=np.float64)

        return result

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        """Convert a float64 ndarray back to a DataFrame.

        Categorical columns are decoded from their ordinal encoding to
        the original values. Numeric columns are cast back to their
        original dtypes.

        Parameters
        ----------
        ``arr`` : np.ndarray
            Array of shape ``(n_rows, n_features)`` or ``(n_features,)``
            as produced by :meth:`transform`.

        Returns
        -------
        pd.DataFrame
            DataFrame with the original column names, dtypes, and
            categorical labels.

        Raises
        ------
        CONFETTIConfigurationError
            If the encoder has not been fitted or if the array has an
            unexpected number of columns.
        """
        columns, dtypes, cat_mappings = self._require_fitted("inverse_transform")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != len(columns):
            raise CONFETTIConfigurationError(
                message=(
                    f"Array has {arr.shape[1]} columns but encoder expects {len(columns)}."
                ),
                param="arr",
                hint="Ensure the array was produced by transform() with the same encoder.",
            )

        data = {}
        for i, col in enumerate(columns):
            if col in cat_mappings:
                mapping = cat_mappings[col]["int_to_value"]
                data[col] = [mapping.get(int(round(v)), v) for v in arr[:, i]]
            else:
                data[col] = arr[:, i].astype(dtypes[col])

        return pd.DataFrame(data)

    def _require_fitted(
        self, method_name: str,
    ) -> tuple[list[str], dict[str, np.dtype], dict[str, dict]]:
        """Return fitted state or raise if the encoder is not fitted.

        Parameters
        ----------
        ``method_name`` : str
            Calling method name, used in the error message.

        Returns
        -------
        tuple[list[str], dict[str, np.dtype], dict[str, dict]]
            ``(columns, dtypes, categorical_mappings)``.

        Raises
        ------
        CONFETTIConfigurationError
            If the encoder has not been fitted.
        """
        if self.columns is None or self.dtypes is None or self.categorical_mappings is None:
            raise CONFETTIConfigurationError(
                message=f"FeatureEncoder.{method_name}() called before fit().",
                param="encoder",
                hint="Call fit() with a reference DataFrame first.",
            )
        return self.columns, self.dtypes, self.categorical_mappings

    def _check_columns(self, df: pd.DataFrame, method_name: str) -> None:
        if list(df.columns) != self.columns:
            raise CONFETTIConfigurationError(
                message=(
                    f"Column mismatch in {method_name}(). "
                    f"Expected {self.columns}, got {list(df.columns)}."
                ),
                param="df",
                hint="Ensure the DataFrame has the same columns as the one used in fit().",
            )
