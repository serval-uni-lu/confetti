from dataclasses import dataclass
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from confetti.errors import CONFETTIError, CONFETTIDataTypeError
from confetti.utils import array_to_string
from pathlib import Path


@dataclass
class Counterfactual():
    """A class to store a single counterfactual explanation results.

    Parameters
    ----------
    counterfactual : np.array
        The counterfactual instance.
    label : str | int | float
        The label of the counterfactual instance.
    """

    counterfactual: np.ndarray
    label: str | int | float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Counterfactual):
            return NotImplemented

        return (
                self.label == other.label and
                np.array_equal(self.counterfactual, other.counterfactual)
        )


class CounterfactualSet():
    """A class to store all counterfactual explanation results for a single instance.

    Parameters
    ----------
    original_instance : np.array
        The original instance for which counterfactuals were generated.
    original_label : str | int | float
        The label of the original instance.
    nearest_unlike_neighbour : np.array
        The nearest unlike neighbour to the original instance.
    best_solution : Counterfactual
        The best counterfactual solution found.
    all_counterfactuals : List[Counterfactual]
        A list of all counterfactuals generated for the original instance.

    Attributes
    ----------
    original_instance : np.ndarray
        The original instance for which counterfactuals were generated.
    nearest_unlike_neighbour : np.ndarray
        The nearest unlike neighbour to the original instance.
    best : Counterfactual
        The best counterfactual solution found.
    all_counterfactuals : List[Counterfactual]
        A list of all counterfactuals generated for the original instance.

    Methods
    -------
    to_dataframe() -> pd.DataFrame
        Export all counterfactuals to a pandas DataFrame.
    output_path: Path | str = Path("./counterfactuals_all_instances.csv")
        Export all counterfactuals to a CSV file.


    """

    def __init__(self,
                 original_instance: np.ndarray,
                 original_label: str | int | float | np.int64 | np.float64,
                 nearest_unlike_neighbour: np.ndarray,
                 best_solution: None |Counterfactual,
                 all_counterfactuals: List[Counterfactual]):

        self._validate_dtypes(original_instance,
                              original_label,
                              nearest_unlike_neighbour,
                              best_solution,
                              all_counterfactuals)

        self._original_instance: np.ndarray = original_instance
        self._original_label: str | int | float = original_label
        self._nearest_unlike_neighbour: np.ndarray = nearest_unlike_neighbour
        self._best: Counterfactual = best_solution
        self._all_counterfactuals: List[Counterfactual] = all_counterfactuals

    @property
    def original_instance(self) -> np.ndarray:
        return self._original_instance

    @property
    def original_label(self) -> Union[str, int, float, np.int64, np.float64]:
        return self._original_label

    @property
    def nearest_unlike_neighbour(self) -> np.ndarray:
        return self._nearest_unlike_neighbour

    @property
    def best(self) -> Counterfactual:
        return self._best

    @property
    def all_counterfactuals(self) -> List[Counterfactual]:
        return self._all_counterfactuals

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all counterfactuals to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all counterfactuals and their details.
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
            }

            df = pd.DataFrame(data)
            return df

    def to_csv(self, output_path: Path | str = Path("./counterfactuals_all_instances.csv")) -> None:
        """
        Export all counterfactuals to a CSV file.

        Parameters
        ----------
        output_directory : str
            The file path where the CSV will be saved.
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
        print(f"Counterfactuals exported to {output_path}")

    @staticmethod
    def _validate_dtypes(
            original_instance: np.ndarray,
            original_label: Union[str, int, float, np.int64, np.float64],
            nearest_unlike_neighbour: np.ndarray,
            best_solution: None | Counterfactual,
            all_counterfactuals: List[Counterfactual]
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


class CounterfactualResults():
    """A class to store counterfactual explanation results for multiple instances.

    Parameters
    ----------
    counterfactual_sets : List[CounterfactualSet]
        A list of CounterfactualSet objects, each representing counterfactuals for a specific instance.

    Attributes
    ----------
    counterfactual_sets : List[CounterfactualSet]
        A list of CounterfactualSet objects containing counterfactuals for multiple instances.

    Methods
    -------
    to_dataframe() -> pd.DataFrame
        Export all counterfactuals for all instances to a pandas DataFrame.
    to_csv(output_path: Path | str = Path("./counterfactuals_all_instances.csv")) -> None
        Export all counterfactuals for all instances to a CSV file.
    """

    def __init__(self, counterfactual_sets: Optional[List[CounterfactualSet]] = None):

        self._counterfactual_sets: None | List[CounterfactualSet] = counterfactual_sets

    @property
    def counterfactual_sets(self) -> List[CounterfactualSet]:
        return self._counterfactual_sets

    def __len__(self) -> int:
        """Return the number of CounterfactualSet objects stored."""
        return len(self._counterfactual_sets or [])

    def __getitem__(self, index: int) -> CounterfactualSet:
        """Return the CounterfactualSet at the specified index."""
        if self._counterfactual_sets is None:
            raise CONFETTIError(
                message="No counterfactual sets available for indexing.",
                param="counterfactual_sets",
                hint="Generate counterfactual sets before attempting to access them."
            )
        return self._counterfactual_sets[index]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all counterfactuals for all instances to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all counterfactuals and their details for all instances.
        """

        if self.counterfactual_sets is None or len(self.counterfactual_sets) == 0:
            raise CONFETTIError(
                message="No counterfactual sets available to export.",
                param="counterfactual_sets",
                hint="Generate counterfactual sets before exporting to DataFrame."
            )
        else:
            return pd.concat([ces.to_dataframe() for ces in self.counterfactual_sets], ignore_index=True)

    def to_csv(self, output_path: Path | str = Path("./counterfactuals.csv") ) -> None:
        """
        Export all counterfactuals for all instances to a CSV file.

        Parameters
        ----------
        output_path : Path
            The file path where the CSV will be saved.
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
