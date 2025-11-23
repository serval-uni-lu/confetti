from dataclasses import dataclass
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from confetti.errors import CONFETTIError, CONFETTIDataTypeError
from confetti.utils import array_to_string
from pathlib import Path


@dataclass
class Counterfactual:
    """Container for a single counterfactual instance and its predicted label.

    Note
    ----
        This class is a lightweight structure used internally by
        :class:`~confetti.explainer.counterfactual_structs.CounterfactualSet`.
    """
    counterfactual: np.ndarray
    """The generated counterfactual time series."""

    label: str | int | float
    """The predicted label corresponding to the counterfactual."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Counterfactual):
            return NotImplemented

        return (
                self.label == other.label and
                np.array_equal(self.counterfactual, other.counterfactual)
        )

class CounterfactualSet:
    """Container for all counterfactual explanations generated for one instance.

    Parameters
    ----------
    original_instance : np.ndarray
        The original instance for which counterfactuals were generated.
    original_label : str | int | float | np.int64 | np.float64
        The predicted label of the original instance.
    nearest_unlike_neighbour : np.ndarray
        The nearest unlike neighbour (NUN) of the original instance.
    best_solution : Counterfactual or None
        The best counterfactual solution identified by the method.
    all_counterfactuals : list of Counterfactual
        All counterfactuals generated for this instance.
    feature_importance : np.ndarray or None, optional
        Optional 1D array of feature-importance values (e.g., CAM weights)
        for the nearest unlike neighbour.


    Note
    ----
        This object stores all counterfactual candidates for a single instance,
        together with metadata such as the NUN and optional importance weights.
        It provides convenience methods for exporting structured results.
    """

    def __init__(self,
                 original_instance: np.ndarray,
                 original_label: str | int | float | np.int64 | np.float64,
                 nearest_unlike_neighbour: np.ndarray,
                 best_solution: None | Counterfactual,
                 all_counterfactuals: List[Counterfactual],
                 feature_importance: Optional[np.ndarray] = None):

        self._validate_dtypes(original_instance,
                              original_label,
                              nearest_unlike_neighbour,
                              best_solution,
                              all_counterfactuals,
                              feature_importance)

        self._original_instance: np.ndarray = original_instance
        self._original_label: str | int | float = original_label
        self._nearest_unlike_neighbour: np.ndarray = nearest_unlike_neighbour
        self._best: Counterfactual = best_solution
        self._all_counterfactuals: List[Counterfactual] = all_counterfactuals
        self._feature_importance: Optional[np.ndarray] = feature_importance  # optional 1D weights

    @property
    def original_instance(self) -> np.ndarray:
        """Return the original instance.

        Returns
        -------
        np.ndarray
            The original instance for which counterfactuals were generated.
        """
        return self._original_instance

    @property
    def original_label(self) -> Union[str, int, float, np.int64, np.float64]:
        """Return the predicted label of the original instance.

        Returns
        -------
        str | int | float | np.int64 | np.float64
            The predicted label.
        """
        return self._original_label

    @property
    def nearest_unlike_neighbour(self) -> np.ndarray:
        """Return the nearest unlike neighbour.

        Returns
        -------
        np.ndarray
            The nearest unlike neighbour instance.
        """
        return self._nearest_unlike_neighbour

    @property
    def best(self) -> Counterfactual:
        """Return the best counterfactual solution.

        Returns
        -------
        Counterfactual
            The best counterfactual, selected according to the
            method's optimization criteria.
        """
        return self._best

    @property
    def all_counterfactuals(self) -> List[Counterfactual]:
        """Return all generated counterfactuals.

        Returns
        -------
        list of Counterfactual
            All counterfactual candidates generated for this instance.
        """
        return self._all_counterfactuals

    @property
    def feature_importance(self) -> Optional[np.ndarray]:
        """Return the optional feature-importance vector.

        Returns
        -------
        np.ndarray or None
            A 1D array of feature-importance weights, or ``None`` if not provided.
        """
        return self._feature_importance

    def to_dataframe(self) -> pd.DataFrame:
        """Return all counterfactuals as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to one counterfactual.
        """

        if self.all_counterfactuals is None or len(self.all_counterfactuals) == 0:
            raise CONFETTIError(
                message="No counterfactuals available to export.",
                param="all_counterfactuals",
                hint="Generate counterfactuals before exporting to DataFrame."
            )
        else:
            data = {
                "counterfactual": [cf.counterfactual for cf in self.all_counterfactuals],
                "is_best": [cf == self.best for cf in self.all_counterfactuals],
                "label": [cf.label for cf in self.all_counterfactuals],
                "original_instance": [self.original_instance] * len(self.all_counterfactuals),
                "original_label": [self.original_label] * len(self.all_counterfactuals),
                "nearest_unlike_neighbour": [self.nearest_unlike_neighbour] * len(self.all_counterfactuals),
                "feature_importance": [self.feature_importance] * len(self.all_counterfactuals)
            }

            df = pd.DataFrame(data)
            return df

    def to_csv(self, output_path: Path | str = Path("./counterfactuals_all_instances.csv")) -> None:
        """Export all counterfactuals to a CSV file.

        Parameters
        ----------
        output_path : str or Path, default="./counterfactuals_all_instances.csv"
            Destination file path for the exported CSV.

        Returns
        -------
        None
        """

        if not isinstance(output_path, (str, Path)):
            raise CONFETTIError(
                message="Invalid output directory for CSV export.",
                param="output_directory",
                hint="Provide a valid string or Path for the output directory."
            )

        df = self.to_dataframe()
        df["counterfactual"] = df["counterfactual"].apply(array_to_string)
        df["original_instance"] = df["original_instance"].apply(array_to_string)
        df["nearest_unlike_neighbour"] = df["nearest_unlike_neighbour"].apply(array_to_string)
        # stringify feature_importance if present
        if "feature_importance" in df.columns and df["feature_importance"].notna().any():
            df["feature_importance"] = df["feature_importance"].apply(
                lambda x: array_to_string(x) if isinstance(x, np.ndarray) else None
            )
        df.to_csv(output_path, index=False)
        print(f"Counterfactuals exported to {output_path}")

    @staticmethod
    def _validate_dtypes(
            original_instance: np.ndarray,
            original_label: Union[str, int, float, np.int64, np.float64],
            nearest_unlike_neighbour: np.ndarray,
            best_solution: None | Counterfactual,
            all_counterfactuals: List[Counterfactual],
            feature_importance: Optional[np.ndarray]
    ) -> None:
        """
        Validate the input types required for counterfactual processing.

        Raises
        ------
        CONFETTIDataTypeError
            If any input does not match its expected type.
        """

        validations = [
            ("original_instance", original_instance, np.ndarray, "numpy array"),
            ("original_label", original_label, (str, int, float, np.int64, np.float64),
             "string, integer, float, np.int64, or np.float64"),
            ("nearest_unlike_neighbour", nearest_unlike_neighbour, np.ndarray, "numpy array"),
        ]

        for param, value, expected_type, expected_str in validations:
            if not isinstance(value, expected_type):
                raise CONFETTIDataTypeError(
                    message=f"{param} must be a {expected_str}.",
                    param=param,
                    hint=f"Ensure that {param} is of type {expected_str}."
                )

        if best_solution is not None and not isinstance(best_solution, Counterfactual):
            raise CONFETTIDataTypeError(
                message="best_solution must be a Counterfactual object or None.",
                param="best_solution",
                hint="Ensure that best_solution is either None or of type Counterfactual."
            )

        if not all(isinstance(cf, Counterfactual) for cf in all_counterfactuals):
            raise CONFETTIDataTypeError(
                message="All entries in all_counterfactuals must be Counterfactual objects.",
                param="all_counterfactuals",
                hint="Ensure that all entries are of type Counterfactual object."
            )

        # new validation for optional feature_importance
        if feature_importance is not None:
            if not isinstance(feature_importance, np.ndarray):
                raise CONFETTIDataTypeError(
                    message="feature_importance must be a numpy array or None.",
                    param="feature_importance",
                    hint="Pass a 1D numpy array of feature weights or leave as None."
                )
            if feature_importance.ndim != 1:
                raise CONFETTIDataTypeError(
                    message="feature_importance must be a 1D numpy array.",
                    param="feature_importance",
                    hint="Reshape or select the appropriate 1D vector of feature weights."
                )


class CounterfactualResults:
    """Container for counterfactual results across multiple instances.

    Parameters
    ----------
    counterfactual_sets : list of CounterfactualSet or None, optional
        A list of :class:`CounterfactualSet` objects, each corresponding to
        one explained instance. If ``None``, the container is initialized empty.

    Note
    ----
        This class acts as a higher-level aggregator over multiple
        :class:`CounterfactualSet` objects. It provides convenience methods
        for exporting structured results for all instances at once.

    """

    def __init__(self, counterfactual_sets: Optional[List[CounterfactualSet]] = None):

        self._counterfactual_sets: None | List[CounterfactualSet] = counterfactual_sets

    @property
    def counterfactual_sets(self) -> List[CounterfactualSet]:
        """Return the list of stored counterfactual sets.

        Returns
        -------
        list of CounterfactualSet
            The stored counterfactual sets.
        """
        return self._counterfactual_sets

    def __len__(self) -> int:
        """Return the number of stored counterfactual sets."""
        return len(self._counterfactual_sets or [])

    def __getitem__(self, index: int) -> CounterfactualSet:
        """Return the counterfactual set at the specified index."""
        if self._counterfactual_sets is None:
            raise CONFETTIError(
                message="No counterfactual sets available for indexing.",
                param="counterfactual_sets",
                hint="Generate counterfactual sets before attempting to access them."
            )
        return self._counterfactual_sets[index]

    def to_dataframe(self) -> pd.DataFrame:
        """Return all counterfactuals across all instances as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all counterfactuals for all instances.
        """

        if self.counterfactual_sets is None or len(self.counterfactual_sets) == 0:
            raise CONFETTIError(
                message="No counterfactual sets available to export.",
                param="counterfactual_sets",
                hint="Generate counterfactual sets before exporting to DataFrame."
            )
        else:
            return pd.concat([ces.to_dataframe() for ces in self.counterfactual_sets], ignore_index=True)

    def to_csv(self, output_path: Path | str = Path("./counterfactuals.csv")) -> None:
        """Export all counterfactuals for all instances to a CSV file.

        Parameters
        ----------
        output_path : str or Path, default="./counterfactuals.csv"
            Destination file path for the exported CSV.

        Returns
        -------
        None
        """
        if not isinstance(output_path, (str, Path)):
            raise CONFETTIError(
                message="Invalid output directory for CSV export.",
                param="output_directory",
                hint="Provide a valid string or Path for the output directory."
            )


        df = self.to_dataframe()
        df["counterfactual"] = df["counterfactual"].apply(array_to_string)
        df["original_instance"] = df["original_instance"].apply(array_to_string)
        df["nearest_unlike_neighbour"] = df["nearest_unlike_neighbour"].apply(array_to_string)
        df.to_csv(output_path, index=False)
        print(f"All counterfactuals exported to {output_path}")

