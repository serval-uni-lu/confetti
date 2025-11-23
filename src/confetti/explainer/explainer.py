from confetti.errors import CONFETTIConfigurationError, CONFETTIError, CONFETTIDataTypeError
from confetti.explainer._counterfactual_problem import CounterfactualProblem
from confetti.structs import Counterfactual, CounterfactualSet, CounterfactualResults

import time
from typing import Optional, List, Tuple
import warnings
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import keras
from tslearn.neighbors import KNeighborsTimeSeries

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination

from multiprocessing import Pool
from functools import partial

class CONFETTI:
    """
    CONFETTI explainer.

    CONFETTI generates counterfactual explanations for multivariate time
    series by combining a nearest-unlike-neighbour search, a naive
    subsequence replacement stage, and a multi-objective optimization stage.
    It supports confidence-based constraints, sparsity control, and optional
    proximity minimization.
    """
    def __init__(self, model_path: Path | str ):
        """
        Initialize a CONFETTI explainer.

        Parameters
        ----------
        model_path : str or Path
            Path to a trained model. Models saved with '.keras'
            are loaded with Keras; files ending in '.joblib' are loaded
            with joblib.
        """
        if not isinstance(model_path, (str,Path)):
            raise CONFETTIConfigurationError(
                message="model_path must be a valid string or Path to the trained model.",
                config={"received_type": type(model_path).__name__},
                param="model_path",
                hint="Pass a valid file path ending in `.keras` or `.joblib`.",
                source="ConfettiExplainer.__init__",
            )
        else:
            self._model_path = model_path
            if str(model_path).endswith(".joblib"):
                self._model = joblib.load(str(model_path))
            else:
                self._model = keras.models.load_model(model_path)

        self._instances_to_explain: Optional[np.ndarray] = None
        self._original_labels: Optional[np.ndarray] = None
        self._reference_data: Optional[np.ndarray] = None
        self._reference_labels: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self.nuns: List[int] = []

    @property
    def model_path(self) -> str:
        """Return the path to the trained model."""
        return self._model_path

    @property
    def model(self):
        """Return the loaded classification model."""
        return self._model

    @property
    def instances_to_explain(self) -> np.ndarray:
        """Instances for which counterfactuals will be generated."""
        return self._instances_to_explain

    @instances_to_explain.setter
    def instances_to_explain(self, instances: np.ndarray) -> None:
        """Set the instances to be explained."""
        if not isinstance(instances, np.ndarray):
            raise CONFETTIDataTypeError(
                message=f"instances_to_explain must be a numpy ndarray, but got {type(instances).__name__} instead.",
                param="instances_to_explain",
                source="instances_to_explain setter"
            )
        self._instances_to_explain = instances

    @property
    def original_labels(self) -> np.ndarray:
        """Original labels of the instances to be explained."""
        return self._original_labels

    @original_labels.setter
    def original_labels(self, labels: np.ndarray) -> None:
        """Set the original labels of the instances to be explained."""
        if not isinstance(labels, np.ndarray):
            raise CONFETTIDataTypeError(
                message=f"original_labels must be a numpy ndarray, but got {type(labels).__name__} instead.",
                param="original_labels",
                source="original_labels setter"
            )
        self._original_labels = labels

    @property
    def reference_data(self) -> np.ndarray:
        """Reference dataset used for finding nearest unlike neighbours."""
        if self._reference_data is None:
            raise CONFETTIConfigurationError(
                message = "Reference data has not been set. Please set reference data before accessing it.",
                param = "reference_data",
                source = "reference_data property"
            )
        return self._reference_data

    @reference_data.setter
    def reference_data(self, data: np.ndarray) -> None:
        """Set the reference dataset used for finding nearest unlike neighbours."""
        if not isinstance(data, np.ndarray):
            raise CONFETTIDataTypeError(
                message=f"reference_data must be a numpy ndarray, but got {type(data).__name__} instead.",
                param="reference_data",
                source="reference_data setter"
            )
        self._reference_data = data

    @property
    def reference_labels(self) -> np.ndarray:
        """Reference labels corresponding to the reference dataset."""
        if self._reference_labels is None:
            raise CONFETTIConfigurationError(
                message = "Reference labels have not been set. Please set reference labels before accessing them.",
                param = "reference_labels",
                source = "reference_labels property"
            )
        return self._reference_labels

    @reference_labels.setter
    def reference_labels(self, labels: np.ndarray) -> None:
        """Set the reference labels corresponding to the reference dataset."""
        if not isinstance(labels, np.ndarray):
            raise CONFETTIDataTypeError(
                message=f"reference_labels must be a numpy ndarray, but got {type(labels).__name__} instead.",
                param="reference_labels",
                source="reference_labels setter"
            )
        self._reference_labels = labels

    @property
    def weights(self) -> None | np.ndarray:
        """Feature importance weights for the reference dataset."""
        return self._weights

    @weights.setter
    def weights(self, weights: None | np.ndarray) -> None:
        """Set the feature importance weights for the reference dataset."""
        if not isinstance(weights, (np.ndarray, type(None))):
            raise CONFETTIDataTypeError(
                message=f"weights must be a numpy ndarray or None, but got {type(weights).__name__} instead.",
                param="weights",
                source="weights setter"
            )
        self._weights = weights

    def _nearest_unlike_neighbour(
        self,
        query: np.ndarray,
        predicted_label: int,
        distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        n_neighbors: int = 1,
        theta: float = 0.51,
    ) -> None | int:
        """
        Find the nearest unlike neighbour (NUN) for a given instance.

        The method restricts the search to reference instances with a
        different predicted label and selects the closest one whose
        predicted confidence meets or exceeds a specified threshold.

        Parameters
        ----------
        query : ndarray of shape (timesteps, channels)
            Instance to be explained.
        predicted_label : int
            Predicted class label of the query.
        distance : str, default="euclidean"
            Distance metric used by KNeighborsTimeSeries.
        dtw_window : int or None, default=None
            Sakoe–Chiba radius when using DTW.
        n_neighbors : int, default=1
            Number of nearest unlike neighbours to retrieve.
        theta : float, default=0.51
            Minimum predicted probability required for a neighbour to
            be considered valid.

        Returns
        -------
        nun_index : int or None
            Index of the nearest unlike neighbour in the reference
            dataset, or None if none meet the confidence threshold.
        """

        # Label DataFrame
        df = pd.DataFrame(self.reference_labels, columns=["label"])
        df.index.name = "index"

        # Only keep unlike-labels
        unlike_mask = df["label"] != predicted_label
        unlike_indices = df[unlike_mask].index
        if self.reference_data is None:
            raise CONFETTIError(
                message="Reference data is not set. Please ensure that reference data is provided before calling this method.",
                param = "reference_data",
                source = "_nearest_unlike_neighbour()"
            )
        X_unlike = self.reference_data[list(unlike_indices)]

        # Fit KNN on unlike instances
        metric_params = None
        if distance == "dtw":
            if dtw_window is not None:
                metric_params = {
                    "global_constraint": "sakoe_chiba",
                    "sakoe_chiba_radius": dtw_window,
                }
        knn = KNeighborsTimeSeries(
            n_neighbors=n_neighbors, metric=distance, metric_params=metric_params
        )
        knn.fit(X_unlike)

        # Get nearest neighbors
        dist, ind = knn.kneighbors(
            query.reshape(1, query.shape[0], query.shape[1]), return_distance=True
        )
        actual_indices = unlike_indices[ind[0]]
        candidate_X = self.reference_data[actual_indices]

        # Use model.predict to get class probabilities
        prob_array = self.model.predict(candidate_X)  # shape: [n_candidates, n_classes]

        # Get predicted confidence (i.e., max prob per candidate)
        confidences = np.max(prob_array, axis=1)

        # Filter by confidence threshold
        keep_mask = confidences >= theta
        if not any(keep_mask):
            return None

        final_indices = actual_indices[keep_mask]

        if len(final_indices) == 0:
            return None
        else:
            nun_index = final_indices[0]
            return nun_index

    def _naive_stage(
        self,
        instance_index: int,
        nun_index: int,
        model,
        subarray_length: int = 1,
        theta: float = 0.51,
        verbose: bool = False,
    ) -> Tuple[Counterfactual, int]:
        """
        Generate a naive counterfactual by subsequence replacement.

        A contiguous subsequence in the original instance is replaced
        with the same subsequence from the nearest unlike neighbour.
        The subsequence length is increased until the resulting
        instance reaches a predicted confidence above a given threshold.

        Parameters
        ----------
        instance_index : int
            Index of the instance to explain.
        nun_index : int
            Index of the nearest unlike neighbour in the reference set.
        model : object
            Trained classifier with a predict method.
        subarray_length : int, default=1
            Initial length of the subsequence to replace.
        theta : float, default=0.51
            Confidence threshold for validity.
        verbose : bool, default=False
            If True, print progress information.

        Returns
        -------
        counterfactual : Counterfactual
            The counterfactual instance produced by the naive stage.
        subarray_length : int
            Subsequence length used when a valid counterfactual was found.
        """

        # Initialize values
        if verbose:
            print(f"Naive stage started for instance {instance_index}")

        if self.weights is None:
            raise CONFETTIError(
                message="It is not possible to perform the naive stage without feature weights.",
                param = "weights",
                source = "_naive_stage()",
                hint = "Please provide feature weights to enable the naive stage."
            )
        else:
            starting_point : int = self._findsubarray(w=self.weights[nun_index], k=subarray_length)

        perturbed_instance : np.ndarray = np.concatenate(
            (
                self.instances_to_explain[instance_index][:starting_point],
                (
                    self.reference_data[nun_index][
                        starting_point : subarray_length + starting_point
                    ]
                ),
                self.instances_to_explain[instance_index][
                    subarray_length + starting_point :
                ],
            )
        )
        perturbed_instance_reshaped = perturbed_instance.reshape(
            1, perturbed_instance.shape[0], perturbed_instance.shape[1]
        )

        prob_target = model.predict(perturbed_instance_reshaped)[0][self.reference_labels[nun_index]]

        while prob_target <= theta:
            subarray_length += 1

            starting_point : int = self._findsubarray(
                w=self.weights[nun_index], k=subarray_length
            )

            perturbed_instance = np.concatenate(
                (
                    self.instances_to_explain[instance_index][:starting_point],
                    (
                        self.reference_data[nun_index][
                            starting_point : subarray_length + starting_point
                        ]
                    ),
                    self.instances_to_explain[instance_index][
                        subarray_length + starting_point :
                    ],
                )
            )

            perturbed_instance_reshaped = perturbed_instance.reshape(
                1, perturbed_instance.shape[0], perturbed_instance.shape[1]
            )

            prob_target = model.predict(perturbed_instance_reshaped)[0][self.reference_labels[nun_index]]

        ce_label = np.argmax(model.predict(perturbed_instance_reshaped),axis=1,)

        if verbose:
            print(f"Naive stage finished for instance {instance_index}")

        counterfactual : Counterfactual = Counterfactual(
            counterfactual=perturbed_instance_reshaped,
            label=ce_label[0],
        )

        return counterfactual, subarray_length

    def _optimization(
        self,
        instance_index: int,
        nun_index: int,
        subsequence_length: int,
        model,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: float = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = True,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        verbose: bool = False,
    ) -> None | CounterfactualSet:
        """
        Optimize a counterfactual using multi-objective evolutionary search.

        The search refines a subsequence determined by the naive stage
        (or the full instance when no naive stage is used). It optimizes
        for a combination of confidence, sparsity, and optionally
        proximity. NSGA-III is used to explore the solution space.

        Parameters
        ----------
        instance_index : int
            Index of the instance to explain.
        nun_index : int
            Index of the nearest unlike neighbour.
        subsequence_length : int
            Initial subsequence length to optimize.
        model : object
            Trained classifier with a predict method.
        alpha : float, default=0.5
            Weight between confidence and sparsity when selecting the
            final counterfactual.
        theta : float, default=0.51
            Minimum predicted probability required for validity.
        n_partitions : int, default=12
            Number of partitions used for NSGA-III reference directions.
        population_size : int, default=100
            Population size for the evolutionary algorithm.
        maximum_number_of_generations : int, default=100
            Maximum number of generations for the search.
        crossover_probability : float, default=1.0
            Probability of crossover.
        mutation_probability : float, default=0.9
            Probability of mutation.
        optimize_confidence : bool, default=True
            Whether to optimize the confidence objective.
        optimize_sparsity : bool, default=True
            Whether to optimize the sparsity objective.
        optimize_proximity : bool, default=True
            Whether to optimize the proximity objective.
        proximity_distance : str, default="euclidean"
            Distance metric used for proximity.
        dtw_window : int or None, default=None
            Sakoe–Chiba radius when using DTW.
        verbose : bool, default=False
            If True, print progress logs.

        Returns
        -------
        counterfactual_set : CounterfactualSet or None
            A set of counterfactuals for the instance, including the
            best solution selected by alpha. None if no valid solutions
            were found.
        """

        # Validate that there are at least two objectives to optimize
        number_of_objectives = (
            int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)
        )
        if number_of_objectives < 2:
            raise CONFETTIConfigurationError(
                "At least two objectives must be optimized. "
                "Set `optimize_confidence`, `optimize_sparsity`, or `optimize_proximity` to True."
            )


        query: np.ndarray = self.instances_to_explain[instance_index]
        nun : np.ndarray = copy.deepcopy(self.reference_data[nun_index])
        all_counterfactuals: List[Counterfactual] = []
        objective_values: List[np.ndarray] = []


        high = subsequence_length
        low = 1
        if verbose:
            print(f"Optimization of CE for Instance {instance_index} started.")

        while low <= high:
            start_time = time.time()
            window = (low + high) // 2
            if verbose:
                print(
                    f"Optimization of CE for Instance {instance_index} in Window {window}"
                )

            # Timestep where it starts
            if self.weights is None:
                starting_point = 0
            else:
                starting_point = self._findsubarray((self.weights[nun_index]), window)
            end_point = starting_point + window

            problem = CounterfactualProblem(
                original_instance=query,
                nun_instance=nun,
                nun_index=nun_index,
                start_timestep=starting_point,
                subsequence_length=window,
                classifier=model,
                reference_labels=self.reference_labels,
                optimize_confidence=optimize_confidence,
                optimize_sparsity=optimize_sparsity,
                optimize_proximity=optimize_proximity,
                proximity_distance=proximity_distance,
                dtw_window=dtw_window,
                theta=theta,
            )

            reference_directions = get_reference_directions(
                name="das-dennis",
                n_dim=number_of_objectives,
                n_partitions=n_partitions
            )

            algorithm = NSGA3(
                pop_size=population_size,
                ref_dirs=reference_directions,
                sampling=BinaryRandomSampling(),
                crossover=TwoPointCrossover(prob=crossover_probability),
                mutation=BitflipMutation(prob=mutation_probability),
            )

            termination = get_termination("n_gen", maximum_number_of_generations)

            result = minimize(problem, algorithm, termination, seed=1, verbose=False)

            if result.X is None:
                low = window + 1
            else:
                objective_values.append(result.F)

                n_samples = result.X.shape[0]
                n_timesteps, n_features = query.shape

                results_reshaped : np.ndarray = result.X.reshape(n_samples, window, n_features)
                perturbations : np.ndarray = np.tile(query, (n_samples, 1, 1))
                replacement_patch : np.ndarray = nun[starting_point:end_point]

                replacement_patch_broadcast : np.ndarray = np.broadcast_to(
                    replacement_patch, (n_samples, window, n_features)
                )

                perturbations[:, starting_point:end_point, :] = np.where(
                    results_reshaped,
                    replacement_patch_broadcast,
                    perturbations[:, starting_point:end_point, :],
                )

                predictions = model.predict(perturbations)
                predicted_labels = np.argmax(predictions, axis=1)


                all_counterfactuals.extend(
                    Counterfactual(counterfactual=cf, label=label)
                    for cf, label in zip(perturbations, predicted_labels)
                )

                high = window - 1
            final_time = time.time() - start_time
            if verbose:
                print(
                    f"Instance: {instance_index} | Window: {window} | Time: {final_time}"
                )
        if verbose:
            print(f"Optimization of CE for Instance {instance_index} finished.")

        if not objective_values:
            return None
        else:
            objectives_array: np.ndarray = np.vstack(objective_values)
            if optimize_confidence and optimize_sparsity:
                best = self._select_best_solution(all_counterfactuals, objectives_array, alpha)
            else:
                best = None

            counterfactuals = CounterfactualSet(
                original_instance=query,
                original_label = self.original_labels[instance_index].item(),
                nearest_unlike_neighbour=nun,
                best_solution = best,
                all_counterfactuals=all_counterfactuals,
                feature_importance=self.weights[nun_index] if self.weights is not None else None
            )

            return counterfactuals

    def _one_pass(
        self,
        test_instance: int,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: float = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = True,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        verbose: bool = False,
    ) -> None | CounterfactualSet:
        """
        Generate a counterfactual for a single instance.

        This method runs the full pipeline for one instance:
        nearest-unlike-neighbour search, naive stage (if weights are
        available), and optimization. It is used internally by both
        sequential and parallel generation.

        Parameters
        ----------
        test_instance : int
            Index of the instance to explain.
        alpha : float, default=0.5
            Weight between confidence and sparsity.
        theta : float, default=0.51
            Confidence threshold.
        n_partitions, population_size, maximum_number_of_generations,
        crossover_probability, mutation_probability :
            Genetic algorithm parameters.
        optimize_confidence, optimize_sparsity, optimize_proximity : bool
            Objective configuration flags.
        proximity_distance : str, default="euclidean"
            Distance metric for proximity objective.
        dtw_window : int or None, default=None
            DTW radius.
        verbose : bool, default=False
            If True, print diagnostic information.

        Returns
        -------
        counterfactual_set : CounterfactualSet or None
            The generated counterfactuals, or None if no valid NUN or
            no counterfactuals were found.
        """

        # Load model
        if self.model_path.endswith(".joblib"):
            model = joblib.load(self.model_path)
        else:
            model = keras.models.load_model(self.model_path)

        nun_index = self._nearest_unlike_neighbour(
            query=self.instances_to_explain[test_instance],
            predicted_label=self.original_labels[test_instance].item(),
            distance=proximity_distance,
            n_neighbors=1,
            theta=theta,
        )

        if nun_index is None:
            if verbose:
                warnings.warn(
                    f"No NUN found for instance {test_instance} with theta {theta}.",
                    UserWarning,
                )
            return None


        if self.weights is None:
            if verbose:
                print("No feature weights were found. Skipping naive stage.")
                print(f"Optimization stage started for instance {test_instance}")
            counterfactual_set: None | CounterfactualSet = self._optimization(
                instance_index=test_instance,
                nun_index=nun_index,
                subsequence_length=self.instances_to_explain.shape[1],
                model=model,
                alpha=alpha,
                theta=theta,
                n_partitions=n_partitions,
                population_size=population_size,
                maximum_number_of_generations=maximum_number_of_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                optimize_confidence=optimize_confidence,
                optimize_sparsity=optimize_sparsity,
                optimize_proximity=optimize_proximity,
                proximity_distance=proximity_distance,
                dtw_window=dtw_window,
                verbose=verbose,
            )
            return counterfactual_set
        else:
            naive, subarray_length = self._naive_stage(
                instance_index=test_instance,
                nun_index=nun_index,
                model=model,
                theta=theta,
                verbose=verbose,
            )
            if verbose:
                print(f"Naive stage finished for instance {test_instance}")
                print(f"Optimization stage started for instance {test_instance}")

            optimized: None | CounterfactualSet = self._optimization(
                instance_index=test_instance,
                nun_index=nun_index,
                subsequence_length=subarray_length,
                model=model,
                alpha=alpha,
                theta=theta,
                n_partitions=n_partitions,
                population_size=population_size,
                maximum_number_of_generations=maximum_number_of_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                optimize_confidence=optimize_confidence,
                optimize_sparsity=optimize_sparsity,
                optimize_proximity=optimize_proximity,
                proximity_distance=proximity_distance,
                dtw_window=dtw_window,
                verbose=verbose,
            )

            if optimized:
                return optimized
            else:
                counterfactual_set = CounterfactualSet(
                    original_instance=self.instances_to_explain[test_instance],
                    original_label=self.original_labels[test_instance].item(),
                    nearest_unlike_neighbour=self.reference_data[nun_index],
                    best_solution=naive,
                    all_counterfactuals=[naive],
                    feature_importance=self.weights[nun_index]
                )
                return counterfactual_set


    def generate_counterfactuals(self,
        instances_to_explain: np.ndarray,
        reference_data: np.ndarray,
        reference_weights: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: int = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = True,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        processes: Optional[int] = None,
        save_counterfactuals: bool = False,
        output_path: Optional[str | Path] = None,
        verbose: bool = False) -> None| CounterfactualResults:
        """
        Generate counterfactual explanations for multiple instances.

        Each instance is processed independently using the CONFETTI
        pipeline. When processes > 1, parallelization is used; otherwise
        the generation proceeds sequentially.

        Parameters
        ----------
        instances_to_explain : ndarray of shape (n_instances, timesteps, channels)
            Instances for which counterfactuals will be generated.
        reference_data : ndarray of shape (n_reference, timesteps, channels)
            Reference dataset used to identify nearest unlike neighbours (NUNs).
        reference_weights : ndarray or None, default=None
            Feature-importance weights used in the naive stage. If None, the naive stage is skipped.
        alpha : float, default=0.5
            Trade-off parameter between confidence and sparsity when selecting
            the best counterfactual.
        theta : float, default=0.51
            Minimum predicted probability required for a counterfactual to be
            considered valid.
        n_partitions : int, default=3
            Number of partitions used to construct the reference directions
            for the NSGA-III algorithm. Larger values create a denser set of
            directions and may increase search resolution at the cost of runtime.
        population_size : int, default=100
            Number of individuals in the evolutionary population. Higher values
            improve the search space exploration but increase computation time.
        maximum_number_of_generations : int, default=100
            Maximum number of evolutionary generations. Increasing this allows
            the genetic algorithm to converge more thoroughly at the expense of
            longer runtimes.
        crossover_probability : float, default=1.0
            Probability of applying crossover during reproduction. A value of
            1.0 means crossover is always applied.
        mutation_probability : float, default=0.9
             Probability of flipping each bit in the binary mask used to select
            perturbed time steps.
        optimize_confidence : bool, default=True
            Whether to include confidence maximization as an optimization
            objective.
        optimize_sparsity : bool, default=True
            Whether to include sparsity minimization as an optimization
            objective.
        optimize_proximity : bool, default=True
            Whether to include proximity minimization as an optimization
            objective.
        proximity_distance : str, default="euclidean"
            Distance metric for proximity. Supported metrics include
            Euclidean and several tslearn-based time-series distances.
        dtw_window : int or None, default=None
            Sakoe–Chiba radius for DTW proximity. If None, no warping
            constraint is applied.
        processes : int or None, default=None
            Number of worker processes for parallel execution. If None,
            counterfactuals are generated sequentially.
        save_counterfactuals : bool, default=False
            Whether to save all generated counterfactuals to CSV files.
        output_path : str or Path or None, default=None
            Directory where CSV outputs should be written. If None, results
            are saved to the current working directory.
        verbose : bool, default=False
            If True, print detailed progress messages during generation.

        Returns
        -------
        counterfactual_results : CounterfactualResults or None
            All counterfactual sets produced for the batch.

        Note
        -----
        Distance metrics for proximity optimization are handled through the
        ``tslearn.metrics`` module when ``optimize_proximity=True``.

        CONFETTI supports several time-series distances beyond Euclidean, including
        Dynamic Time Warping (DTW), Soft-DTW, CTW, and GAK. These metrics are accessed
        internally via ``tslearn.metrics`` through functions such as
        ``cdist_dtw``, ``cdist_ctw``, ``cdist_soft_dtw`` and ``cdist_gak``.

        - If a tslearn-based distance is selected (e.g., ``proximity_distance="dtw"``),
          the proximity objective computes pairwise distances between each generated
          counterfactual and the original instance using the corresponding tslearn
          cost matrix.
        - DTW can optionally use a Sakoe–Chiba band via ``dtw_window`` for faster and
          more constrained alignment.
        - ``tslearn`` is **only required** when a non-Euclidean proximity metric is
          requested. If tslearn is not installed and such a metric is selected,
          CONFETTI raises a ``CONFETTIConfigurationError`` describing the missing
          dependency.

        Euclidean proximity does not require tslearn and is always available.
        """

        if processes is not None:
            self._validate_types(locals(), context="parallelization")
        else:
            self._validate_types(locals())

        self._instances_to_explain : np.ndarray = instances_to_explain
        self._original_labels : np.ndarray = np.argmax(self.model.predict(self.instances_to_explain), axis=1)
        self.reference_data : np.ndarray = reference_data
        self.reference_labels : np.ndarray = np.argmax(self.model.predict(self.reference_data), axis=1)
        self.weights = reference_weights

        results : None | CounterfactualResults = None

        if processes is not None:
            results = self._parallelized_generator(
                alpha=alpha,
                theta=theta,
                n_partitions=n_partitions,
                population_size=population_size,
                maximum_number_of_generations=maximum_number_of_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                optimize_confidence=optimize_confidence,
                optimize_sparsity=optimize_sparsity,
                optimize_proximity=optimize_proximity,
                proximity_distance=proximity_distance,
                dtw_window=dtw_window,
                processes=processes,
                verbose=verbose,
            )
        else:
            results = self._generator(
                instances_to_explain=instances_to_explain,
                alpha=alpha,
                theta=theta,
                n_partitions=n_partitions,
                population_size=population_size,
                maximum_number_of_generations=maximum_number_of_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                optimize_confidence=optimize_confidence,
                optimize_sparsity=optimize_sparsity,
                optimize_proximity=optimize_proximity,
                proximity_distance=proximity_distance,
                dtw_window=dtw_window,
                verbose=verbose,
            )

        if results is None:
            if verbose:
                print("No counterfactuals were generated.")
            return None
        else:
            if save_counterfactuals:
                if output_path is not None:
                    output_path = Path(output_path)
                    output_path.mkdir(parents=True, exist_ok=True)
                    results.to_csv(output_path)
                else:
                    results.to_csv()

            return results


    def _parallelized_generator(
        self,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: int = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = True,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        processes: int = 8,
        verbose: bool = False,
    ) -> None | CounterfactualResults:
        """
        Generate counterfactuals in parallel.

        Each instance is processed through the full CONFETTI pipeline
        in an independent process. Results are aggregated into a single
        CounterfactualResults object.

        Parameters
        ----------
        (same as generate_counterfactuals, except processes is required)

        Returns
        -------
        counterfactual_results : CounterfactualResults or None
            All counterfactual sets from parallel execution.
        """

        pool = Pool(processes=processes)

        wrapped_one_pass = partial(
            self._one_pass,
            alpha=alpha,
            theta=theta,
            n_partitions=n_partitions,
            population_size=population_size,
            maximum_number_of_generations=maximum_number_of_generations,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            optimize_confidence=optimize_confidence,
            optimize_sparsity=optimize_sparsity,
            optimize_proximity=optimize_proximity,
            proximity_distance=proximity_distance,
            dtw_window=dtw_window,
            verbose=verbose,
        )

        results = pool.map(wrapped_one_pass, range(len(self.instances_to_explain)))
        pool.close()
        pool.join()

        self.counterfactual_sets : List[CounterfactualSet]  = []

        for r in results:
            counterfactual_set : CounterfactualSet = r
            #self.nuns.append(nun)
            if counterfactual_set is not None:
                self.counterfactual_sets.append(counterfactual_set)

        if self.counterfactual_sets:
            return CounterfactualResults(counterfactual_sets=self.counterfactual_sets)
        else:
            return None


    def _generator(
        self,
        instances_to_explain: np.ndarray,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: int = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = True,
        proximity_distance: str = "euclidean",
        dtw_window: Optional[int] = None,
        verbose: bool = False,
    ) -> None | CounterfactualResults:

        """
        Generate counterfactuals sequentially.

        This version processes instances one-by-one without spawning
        additional processes. It is the fallback when parallelization
        is not requested.

        Parameters
        ----------
        (subset of generate_counterfactuals parameters)

        Returns
        -------
        counterfactual_results : CounterfactualResults or None
            All counterfactual sets generated sequentially.
        """
        counterfactual_sets: List[CounterfactualSet] = []

        for instance_index in range(len(instances_to_explain)):

            nun_index = self._nearest_unlike_neighbour(
                query=self.instances_to_explain[instance_index],
                predicted_label=self.original_labels[instance_index].item(),
                distance=proximity_distance,
                n_neighbors=1,
                theta=theta,
            )
            if nun_index is None:
                if verbose:
                    print(f"Skipping instance {instance_index}: No valid NUN found.")
                continue


            if self.weights is not None:
                naive, subarray_length = self._naive_stage(
                    instance_index=instance_index,
                    nun_index=nun_index,
                    model=self.model,
                    theta=theta,
                )

                if naive is None:
                    if verbose:
                        print(f"Skipping instance {instance_index}: No valid naive counterfactual found.")
                    continue

                optimized: None | CounterfactualSet = self._optimization(
                    instance_index=instance_index,
                    nun_index=nun_index,
                    subsequence_length=subarray_length,
                    model=self.model,
                    alpha=alpha,
                    theta=theta,
                    n_partitions=n_partitions,
                    population_size=population_size,
                    maximum_number_of_generations=maximum_number_of_generations,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    optimize_confidence=optimize_confidence,
                    optimize_sparsity=optimize_sparsity,
                    optimize_proximity=optimize_proximity,
                    proximity_distance=proximity_distance,
                    dtw_window=dtw_window,
                    verbose=verbose,
                )
                if optimized is not None:
                    counterfactual_sets.append(optimized)
                else:
                    counterfactual_set = CounterfactualSet(
                        original_instance=self.instances_to_explain[instance_index],
                        original_label=self.original_labels[instance_index].item(),
                        nearest_unlike_neighbour=self.reference_data[nun_index],
                        best_solution=naive,
                        all_counterfactuals=[naive],
                    )
                    counterfactual_sets.append(counterfactual_set)
            else:
                if verbose:
                    print("Skipping Naive Stage as no weights were provided.")

                optimized: None | CounterfactualSet = self._optimization(
                    instance_index=instance_index,
                    nun_index=nun_index,
                    subsequence_length=self.instances_to_explain.shape[1],
                    model=self.model,
                    alpha=alpha,
                    theta=theta,
                    n_partitions=n_partitions,
                    population_size=population_size,
                    maximum_number_of_generations=maximum_number_of_generations,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    optimize_confidence=optimize_confidence,
                    optimize_sparsity=optimize_sparsity,
                    optimize_proximity=optimize_proximity,
                    proximity_distance=proximity_distance,
                    dtw_window=dtw_window,
                    verbose=verbose,
                )
                if optimized is not None:
                    counterfactual_sets.append(optimized)
                else:
                    if verbose:
                        print(f"No valid optimized counterfactual found for instance {instance_index}.")
                    continue

        if counterfactual_sets:
            return CounterfactualResults(
                counterfactual_sets=counterfactual_sets)
        else:
            return None

    @staticmethod
    def _findsubarray(w: np.ndarray, k: int) -> int:
        """
        Identify the starting index of the contiguous subsequence of length ``k`` in the list ``w``
        that has the maximum total sum.

        This method performs a linear scan using a sliding window approach to efficiently find
        the most "important" region of the input signal, where ``w`` represents feature attribution weights
         (e.g. Class Activation Map weights)

        Parameters:
        ----------
        ``w`` : list
            A list of importance weights (e.g., CAM values) for each time step of the NUN.
        ``k`` : int
            Desired length of the subsequence.

        Returns:
        -------
        int
            Starting index of the contiguous subsequence of length `k` with the highest total weight.
        """
        start = 0
        max_sum = curr_sum = sum(w[start: start + k])
        for i in range(1, len(w) - k + 1):
            curr_sum -= w[i - 1]
            curr_sum += w[i + k - 1]
            if curr_sum > max_sum:
                max_sum = curr_sum
                start = i
        return start

    @staticmethod
    def _select_best_solution(
            counterfactuals: List[Counterfactual],
            objective_values: np.ndarray,
            alpha: int | float
    ) -> Counterfactual:
        """Select the most relevant counterfactual from a counterfactual set given user weights.

        The function applies a weighted sum using the provided weights (which are normalized to sum to 1 if positive).

        Parameters
        ----------
        counterfactuals : CounterfactualSet
            A counterfactual set including all counterfactuals obtained from the optimization stage.
        objective_values : np.ndarray, shape = (n_solutions, n_objectives)
            Objective values for each counterfactual solution. Each row corresponds to a solution,
        alpha : int | float
            Weighting parameter between confidence and sparsity in the objective function.
            The higher the value, the more weight is given to confidence.

        Returns
        -------
        best : Counterfactual
            The counterfactual solution with the highest weighted score.

        Note
        -----
        Confidence is maximized while sparsity is minimized. However, in Pymoo's implementation, to maximize
        an objective you need to set it to negative. (e.g. f1 = -confidence, f2 = sparsity). Thus, confidence in
        objective values will appear as negative while sparsity as positive.
        Therefore, it is necessary to first flip confidence so both objectives are higher-is-better before applying weights.

        """

        objective_values_flipped = objective_values[:, :2]
        objective_values_flipped[:, 0] = -objective_values_flipped[:, 0]

        weights = np.array([alpha, 1 - alpha])

        scores = np.dot(objective_values_flipped, weights)
        best_index = np.argmax(scores)
        best = counterfactuals[best_index]

        return best

    @staticmethod
    def _validate_types(arguments: dict, context: Optional[str] = None) -> None:
        """
        Validates argument types for CONFETTI generators. Handles both parallel and non-parallel modes.
        """
        # Shared arguments
        expected_types = {
            "instances_to_explain": np.ndarray,
            "reference_data": np.ndarray,
            "reference_weights": (np.ndarray, type(None)),
            "alpha": float,
            "theta": float,
            "n_partitions": int,
            "population_size": int,
            "maximum_number_of_generations": int,
            "crossover_probability": float,
            "mutation_probability": float,
            "optimize_confidence": bool,
            "optimize_sparsity": bool,
            "optimize_proximity": bool,
            "proximity_distance": str,
            "dtw_window": (int, type(None)),
            "save_counterfactuals": bool,
            "verbose": bool,
        }

        # Add only if using the parallel version
        if context == "parallelization":
            expected_types["processes"] = int

        for name, expected in expected_types.items():
            value = arguments.get(name)
            if not isinstance(value, expected):
                expected_name = (
                    ", ".join(t.__name__ for t in expected)
                    if isinstance(expected, tuple)
                    else expected.__name__
                )
                raise CONFETTIConfigurationError(
                    f"{context + ': ' if context else ''}"
                    f"Parameter `{name}` must be of type {expected_name}, "
                    f"but got {type(value).__name__}."
                )

        output_path = arguments.get("output_path")
        save = arguments.get("save_counterfactuals")
        if save:
            if output_path is not None and not isinstance(output_path, (str, Path)):
                raise CONFETTIConfigurationError(
                    f"{context + ': ' if context else ''}"
                    "`output_path` must be a None or a valid string."
                )