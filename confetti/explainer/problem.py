from pymoo.core.problem import Problem
from typing import Optional
import numpy as np
from confetti.explainer.errors import CONFETTIConfigurationError
from tslearn.metrics import TSLEARN_VALID_METRICS
import importlib

# TODO: Delete prints in proximity function.


class CounterfactualProblem(Problem):
    def __init__(
        self,
        original_instance: np.array,
        nun_instance: np.array,
        nun_index: int,
        start_timestep: int,
        subsequence_length: int,
        classifier,
        reference_labels: np.array,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = False,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        alpha: float = 0.5,
        theta: float = 0.51,
    ):
        """
        Define a multi-objective optimization problem for generating counterfactuals
        using a perturbed subsequence of the original time series instance.

        This class is designed to be used with evolutionary algorithms from pymoo. It encodes
        the counterfactual generation process as a two-objective optimization problem, where
        the goals are to maximize confidence in the target class and minimize the number of changes (sparsity).
        A third constraint ensures that the predicted probability exceeds a given threshold (`theta`).

        Parameters:
        ----------
        ``original_instance`` : np.array
            The original multivariate time series instance to be explained.
            Shape: (timesteps, channels)
        ``nun_instance`` : np.array
            The Nearest Unlike Neighbour (NUN) used to guide counterfactual generation.
        ``nun_index`` : int
            Index of the NUN in the reference dataset.
        ``start_timestep`` : int
            Starting index of the subsequence to be perturbed.
        ``subsequence_length`` : int
            Length of the subsequence within the time series to be modified.
        ``classifier`` : object
            Trained classifier with a `predict_proba` method.
        ``reference_labels`` : np.array
            Labels for the reference dataset, used to evaluate class change.
        ``optimize_confidence`` : bool, default=True
            If True, the optimization will add the objective of achieving a confident prediction for the target class.
        ``optimize_sparsity`` : bool, default=True
            If True, the optimization will add the objective of minimizing the number of perturbed time steps.
        ``optimize_proximity`` : bool, default=False
            If True, the optimization will add the objective of minimizing the distance to the original instance.
        ``proximity_distance`` : str, default="euclidean"
            Distance metric to use for proximity optimization. Only used if `optimize_proximity` is True.
        ``dtw_window`` : Optional[int], default=None
            Sakoe–Chiba band radius for DTW. If None, no band constraint is applied.
        ``alpha`` : float, default=0.5
            Trade-off parameter between sparsity and confidence.
            A higher value puts more weight on achieving confident predictions.
            It is not used if `optimize_confidence` or `optimize_sparsity` is False.
        ``theta`` : float, default=0.51
            Confidence threshold for the target class (i.e., predicted probability must be ≥ theta).

        Attributes:
        ----------
        n_var : int
            Number of decision variables (equal to channels × subsequence_length).
        n_obj : int
            Number of objectives (confidence, sparsity, proximity).
        n_ieq_constr : int
            Number of inequality constraints (e.g., confidence ≥ theta).
        """

        self.original_instance = original_instance
        self.nun_instance = nun_instance
        self.nun_index = nun_index
        self.start_timestep = start_timestep
        self.end_timestep = start_timestep + subsequence_length
        self.classifier = classifier
        self.reference_labels = reference_labels
        self.subsequence_length = subsequence_length
        self.optimize_confidence = optimize_confidence
        self.optimize_sparsity = optimize_sparsity
        self.optimize_proximity = optimize_proximity
        self.proximity_distance = proximity_distance
        self.dtw_window = dtw_window
        self.alpha = alpha
        self.theta = theta
        n_var: int = original_instance.shape[1] * (
            self.subsequence_length
        )  # Here, shape[1] are the no. of channels
        n_obj: int = (
            int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)
        )

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Here, x represents a batch of possible counterfactual explanations
        """
        n_samples: int = x.shape[0]
        n_features: int = self.original_instance.shape[1]
        # Reshape the mask to (n_samples, subsequence_length, n_features)
        x_reshaped = x.reshape(n_samples, self.subsequence_length, n_features)

        # Repeat original instance across all samples
        counterfactuals = np.tile(self.original_instance, (n_samples, 1, 1))

        # Extract the patch to replace from nun_instance (shared for all samples)
        replacement_patch = self.nun_instance[
            self.start_timestep : self.end_timestep
        ]  # shape: (subsequence_length, n_features)

        # Broadcast replacement_patch to match number of samples
        replacement_patch_broadcasted = np.broadcast_to(
            replacement_patch, (n_samples, self.subsequence_length, n_features)
        )

        # Apply mask-based replacement
        counterfactuals[:, self.start_timestep : self.end_timestep, :] = np.where(
            x_reshaped,
            replacement_patch_broadcasted,
            counterfactuals[:, self.start_timestep : self.end_timestep, :],
        )

        f1, f2, f3 = None, None, None

        f1 = self.classifier.predict(counterfactuals)[
            :, self.reference_labels[self.nun_index]
        ]
        out["G"] = [self.theta - f1]

        if self.optimize_sparsity:
            f2 = np.sum(x_reshaped, axis=(1, 2))

        if self.optimize_proximity:
            f3 = self._proximity(
                counterfactuals=counterfactuals,
                original_instance=self.original_instance,
                metric=self.proximity_distance,
                dtw_window=self.dtw_window,
            )

        objective_values = []
        if self.optimize_confidence and self.optimize_sparsity:
            objective_values.append(-self.alpha * f1)
            objective_values.append((1.0 - self.alpha) * f2)  # sparsity
        else:
            if self.optimize_confidence:
                objective_values.append(-f1)
            if self.optimize_sparsity:
                objective_values.append(f2)

        if self.optimize_proximity:
            objective_values.append(f3)

        out["F"] = np.row_stack(objective_values).T

    def _proximity(
        self,
        counterfactuals: np.ndarray,
        original_instance: np.ndarray,
        metric: str = "euclidean",
        dtw_window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculate the proximity of the counterfactuals to the original instance
        using the specified distance metric.

        Parameters:
        ----------
        ``counterfactuals`` : np.ndarray
            Batch of counterfactuals. Shape: (N, T, C). The generated counterfactuals to evaluate proximity
            against the original instance.
        ``original_instance`` : np.ndarray
            The original instance to which the counterfactuals are compared.
            Shape: (T, C). The original time series instance.
        ``metric`` : str, default="euclidean"
            The distance metric to use for proximity calculation. Supported options are "euclidean", "manhattan", "dtw".
        ``dtw_window`` : Optional[int], default=None
            Sakoe–Chiba band radius for DTW. If None, no band constraint.

        Returns:
        -------
            np.ndarray
            Proximity values for each counterfactual. Shape: (N,).
        """
        metric_name = metric.lower()
        if metric_name == "euclidean":
            return np.sqrt(
                np.sum(
                    (counterfactuals - original_instance[None, :, :]) ** 2, axis=(1, 2)
                )
            )
        else:
            tslearn_distance_function = self._select_tslearn_distance_function(
                metric_name=metric_name
            )

        X = counterfactuals  # (N, T, C)
        Y = original_instance[None, :, :]  # (1, T, C)

        if metric_name == "dtw" and dtw_window is not None and dtw_window > 0:
            distances = tslearn_distance_function(
                X,
                Y,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=int(dtw_window),
            )
        else:
            distances = tslearn_distance_function(X, Y)

        return distances.ravel()

    def _select_tslearn_distance_function(self, metric_name: str = "dtw"):
        """
        Select the appropriate distance function from tslearn based on the specified metric.
        Args:
            metric_name: The name of the distance metric to use.

        Returns:
            function: The selected distance function.
        """
        try:
            tslearn_metrics = importlib.import_module("tslearn.metrics")
        except ImportError as e:
            raise ImportError(
                "Distance metrics from tslearn require tslearn. Install with `pip install tslearn`."
            ) from e

        name_map = {
            "dtw": "cdist_dtw",
            "ctw": "cdist_ctw",
            "softdtw": "cdist_soft_dtw",
            "gak": "cdist_gak",
        }
        cdist_function_name = name_map.get(metric_name.lower())

        try:
            cdist_function = getattr(tslearn_metrics, cdist_function_name)
        except AttributeError as e:
            raise AttributeError(
                f"tslearn does not provide a distance function for metric '{metric_name}'."
            ) from e

        return cdist_function
