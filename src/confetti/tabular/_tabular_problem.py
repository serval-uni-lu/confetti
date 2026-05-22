from __future__ import annotations

import numpy as np

from confetti.algorithm._problem import Problem
from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError

_SUPPORTED_METRICS = {"euclidean", "manhattan"}


class TabularCounterfactualProblem(Problem):
    """Multi-objective optimization problem for tabular counterfactuals.

    Each decision variable is a binary flag indicating whether the
    corresponding feature value should be replaced with the value from
    the nearest unlike neighbour (NUN).  The problem optimizes a
    combination of confidence, sparsity, and proximity subject to a
    minimum-confidence constraint.

    Parameters
    ----------
    ``original_instance`` : np.ndarray
        Feature vector of the instance to explain.
        Shape: ``(n_features,)``.
    ``nun_instance`` : np.ndarray
        Feature vector of the nearest unlike neighbour.
        Shape: ``(n_features,)``.
    ``nun_index`` : int
        Index of the NUN in the reference dataset.
    ``classifier`` : object
        Trained classifier exposing a ``predict`` method that returns
        class probabilities of shape ``(n_samples, n_classes)``.
    ``reference_labels`` : np.ndarray
        Predicted labels for the reference dataset.
    ``optimize_confidence`` : bool, default=True
        Include confidence maximization as an objective.
    ``optimize_sparsity`` : bool, default=True
        Include sparsity minimization as an objective.
    ``optimize_proximity`` : bool, default=False
        Include proximity minimization as an objective.
    ``proximity_distance`` : str, default="euclidean"
        Distance metric for proximity. Must be ``"euclidean"`` or
        ``"manhattan"``.
    ``theta`` : float, default=0.51
        Minimum predicted probability for the target class.

    Attributes
    ----------
    n_var : int
        Number of decision variables (equal to ``n_features``).
    n_obj : int
        Number of objectives.
    n_ieq_constr : int
        Number of inequality constraints (always 1).

    Raises
    ------
    CONFETTIDataTypeError
        If instance arrays are not 1-D or have mismatched shapes.
    CONFETTIConfigurationError
        If fewer than two objectives are selected, if the proximity
        metric is unsupported, or if *theta* is out of range.
    """

    def __init__(
        self,
        original_instance: np.ndarray,
        nun_instance: np.ndarray,
        nun_index: int,
        classifier,
        reference_labels: np.ndarray,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = False,
        proximity_distance: str = "euclidean",
        theta: float = 0.51,
    ):
        self._validate_init(
            original_instance,
            nun_instance,
            optimize_confidence,
            optimize_sparsity,
            optimize_proximity,
            proximity_distance,
            theta,
        )

        self.original_instance = original_instance
        self.nun_instance = nun_instance
        self.nun_index = nun_index
        self.classifier = classifier
        self.reference_labels = reference_labels
        self.optimize_confidence = optimize_confidence
        self.optimize_sparsity = optimize_sparsity
        self.optimize_proximity = optimize_proximity
        self.proximity_distance = proximity_distance
        self.theta = theta

        n_var = original_instance.shape[0]
        n_obj = int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a batch of binary masks against all objectives.

        Parameters
        ----------
        ``x`` : np.ndarray
            Binary decision matrix of shape ``(n_samples, n_features)``.
        ``out`` : dict
            Output dictionary populated with ``"F"`` (objectives) and
            ``"G"`` (constraints).
        """
        n_features = self.original_instance.shape[0]
        counterfactuals = np.where(x, self.nun_instance, self.original_instance)

        target_label = self.reference_labels[self.nun_index]
        f1 = self.classifier.predict(counterfactuals)[:, target_label]

        out["G"] = [self.theta - f1]

        f2, f3 = None, None

        if self.optimize_sparsity:
            f2 = np.sum(x, axis=1) / n_features

        if self.optimize_proximity:
            f3 = self._proximity(counterfactuals, self.original_instance, self.proximity_distance)

        objective_values = []
        if self.optimize_confidence and self.optimize_sparsity:
            objective_values.append(f1)
            objective_values.append(f2)
        else:
            if self.optimize_confidence:
                objective_values.append(-f1)
            if self.optimize_sparsity:
                objective_values.append(f2)

        if self.optimize_proximity:
            objective_values.append(f3)

        out["F"] = np.vstack(objective_values).T

    @staticmethod
    def _proximity(
        counterfactuals: np.ndarray,
        original_instance: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Compute distance between counterfactuals and the original.

        Parameters
        ----------
        ``counterfactuals`` : np.ndarray
            Batch of counterfactuals, shape ``(n_samples, n_features)``.
        ``original_instance`` : np.ndarray
            Original feature vector, shape ``(n_features,)``.
        ``metric`` : str
            ``"euclidean"`` or ``"manhattan"``.

        Returns
        -------
        np.ndarray
            Distance per counterfactual, shape ``(n_samples,)``.
        """
        diff = counterfactuals - original_instance
        if metric == "euclidean":
            return np.sqrt(np.sum(diff**2, axis=1))
        return np.sum(np.abs(diff), axis=1)

    @staticmethod
    def _validate_init(
        original_instance: np.ndarray,
        nun_instance: np.ndarray,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        theta: float,
    ) -> None:
        """Validate constructor arguments.

        Raises
        ------
        CONFETTIDataTypeError
            If instance arrays are not 1-D or shapes do not match.
        CONFETTIConfigurationError
            If fewer than two objectives, unsupported metric, or theta
            out of range.
        """
        if original_instance.ndim != 1:
            raise CONFETTIDataTypeError(
                message=f"original_instance must be 1-D, got {original_instance.ndim}-D.",
                param="original_instance",
                hint="Pass a feature vector of shape (n_features,).",
            )
        if nun_instance.ndim != 1:
            raise CONFETTIDataTypeError(
                message=f"nun_instance must be 1-D, got {nun_instance.ndim}-D.",
                param="nun_instance",
                hint="Pass a feature vector of shape (n_features,).",
            )
        if original_instance.shape != nun_instance.shape:
            raise CONFETTIConfigurationError(
                message=(
                    f"Shape mismatch: original_instance {original_instance.shape} vs nun_instance {nun_instance.shape}."
                ),
                param="nun_instance",
                hint="Both instances must have the same number of features.",
            )

        n_obj = int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)
        if n_obj < 2:
            raise CONFETTIConfigurationError(
                message="At least two objectives must be enabled.",
                hint="Set at least two of optimize_confidence, optimize_sparsity, optimize_proximity to True.",
            )

        if proximity_distance not in _SUPPORTED_METRICS:
            raise CONFETTIConfigurationError(
                message=f"Unsupported proximity metric '{proximity_distance}' for tabular data.",
                param="proximity_distance",
                hint=f"Choose from {sorted(_SUPPORTED_METRICS)}.",
            )

        if not (0 < theta < 1):
            raise CONFETTIConfigurationError(
                message=f"theta must be in (0, 1), got {theta}.",
                param="theta",
                hint="Use a value like 0.51.",
            )
