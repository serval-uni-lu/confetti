import time
from typing import Optional, List
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

from tslearn.neighbors import KNeighborsTimeSeries

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
from .problem import CounterfactualProblem

from multiprocessing import Pool
from functools import partial

from .utils import convert_string_to_array, array_to_string
from .errors import CONFETTIConfigurationError


class CONFETTI:
    def __init__(self, model_path:str = None):
        """
        Initialize the CONFETTI explainer with the model and data.
        Args:
            model_path: Path to the trained model
        """
        if model_path is None or not isinstance(model_path, str):
            raise CONFETTIConfigurationError(f"model_path must be a valid string path to the trained model, "
                                             f"but got {type(model_path).__name__} instead.")

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        self.instances_to_explain: Optional[np.array] = None
        self.original_labels: Optional[np.array] = None
        self.reference_data: Optional[np.array] = None
        self.reference_labels: Optional[np.array] = None
        self.weights: Optional[np.array] = None
        self.nuns: List[int] = []

    @staticmethod
    def findsubarray(w: list, k: int) -> int:
        """
        Identify the starting index of the contiguous subsequence of length ``k`` in the list ``w``
        that has the maximum total sum.

        This method performs a linear scan using a sliding window approach to efficiently find
        the most "important" region of the input signal, where ``w`` typically represents class
        activation map (CAM) weights.

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
        max_sum = curr_sum = sum(w[start:start + k])
        for i in range(1, len(w) - k + 1):
            curr_sum -= w[i - 1]
            curr_sum += w[i + k - 1]
            if curr_sum > max_sum:
                max_sum = curr_sum
                start = i
        return start

    def nearest_unlike_neighbour(self,
                                 query: np.array,
                                 predicted_label: int,
                                 distance: str = "euclidean",
                                 n_neighbors: int = 1,
                                 theta: float = 0.51
                                 ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find the Nearest Unlike Neighbour (NUN) of a given instance based on a distance metric
        and a minimum confidence constraint.

        This method retrieves the closest instances from the reference set that have a different
        predicted class label than the query instance and whose predicted confidence exceeds
        a given threshold ``theta``. The search is performed using a k-nearest neighbors approach
        restricted to unlike-label instances.

        Parameters:
        ----------
        query : np.array
            The instance to be explained, shaped (timesteps, dimensions).
        predicted_label : int
            The predicted class label of the query instance (e.g., obtained via `np.argmax(model.predict(...))`).
        distance : str, default="euclidean"
            Distance metric to use for k-NN search. Must be compatible with `KNeighborsTimeSeries`.
        n_neighbors : int, default=1
            Number of nearest neighbors to retrieve.
        theta : float, default=0.51
            Minimum predicted probability required for a neighbor to be considered valid.

        Returns:
        -------
        tuple[Optional[np.ndarray], Optional[np.ndarray]]
            - distances : np.ndarray of shape (≤ n_neighbors,)
                Distances from the query to valid unlike neighbors.
            - indices : np.ndarray of shape (≤ n_neighbors,)
                Indices in the reference dataset of the valid unlike neighbors.

            Returns (None, None) if no unlike neighbors meet the confidence threshold.
        """

        # Label DataFrame
        df = pd.DataFrame(self.reference_labels, columns=['label'])
        df.index.name = 'index'

        # Only keep unlike-labels
        unlike_mask = df['label'] != predicted_label
        unlike_indices = df[unlike_mask].index
        X_unlike = self.reference_data[list(unlike_indices)]

        # Fit KNN on unlike instances
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(X_unlike)

        # Get nearest neighbors
        dist, ind = knn.kneighbors(query.reshape(1, query.shape[0], query.shape[1]), return_distance=True)
        actual_indices = unlike_indices[ind[0]]
        candidate_X = self.reference_data[actual_indices]

        # Use model.predict to get class probabilities
        prob_array = self.model.predict(candidate_X)  # shape: [n_candidates, n_classes]

        # Get predicted confidence (i.e., max prob per candidate)
        confidences = np.max(prob_array, axis=1)

        # Filter by confidence threshold
        keep_mask = confidences >= theta
        if not any(keep_mask):
            return None, None

        # Return filtered distances and corresponding indices
        final_distances = dist[0][keep_mask]
        final_indices = actual_indices[keep_mask]

        return final_distances, final_indices

    def naive_stage(self,
                       instance_index: int,
                       nun_index: int,
                       model,
                       subarray_length: int = 1,
                       theta: float = 0.51,
                       verbose: bool = False) -> pd.DataFrame:
        """
        Generate a naive counterfactual by replacing a subsequence of the original instance
        with the corresponding subsequence from its nearest unlike neighbor (NUN).

        This method does not perform any optimization. It directly swaps a fixed-length
        subsequence in the original instance with the same position from the NUN and checks
        whether the resulting instance reaches the desired prediction confidence (``theta``).

        Parameters:
        ----------
        ``instance_index`` : int
            Index of the original instance in the sample dataset.
        ``nun_index`` : int
            Index of the nearest unlike neighbor (NUN) in the reference dataset.
        ``model`` : object
            Trained classification model with a `predict_proba` method.
        ``subarray_length`` : int, default=1
            Length of the subsequence (temporal window) to replace in the original instance.
        ``theta`` : float, default=0.51
            Minimum prediction probability threshold required for the counterfactual to be considered valid.
        ``verbose`` : bool, default=False
            If True, prints detailed information during execution.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the counterfactual solution, window size, test instance index,
            NUN instance index, and the predicted class label of the counterfactual.
        """

        # Initialize values
        if verbose:
            print(f'Naive stage started for instance {instance_index}')

        starting_point = self.findsubarray(w=self.weights[nun_index], k=subarray_length)

        counterfactual = np.concatenate((self.instances_to_explain[instance_index][:starting_point],
                                         (self.reference_data[nun_index][starting_point:subarray_length + starting_point]),
                                         self.instances_to_explain[instance_index][subarray_length + starting_point:]))

        #Obtain prediction probability of target class
        prob_target = model.predict(counterfactual.reshape(1, counterfactual.shape[0], counterfactual.shape[1]), verbose=0)[0][
            self.reference_labels[nun_index]]

        while prob_target <= theta:
            subarray_length += 1
            # Timestep where it starts
            starting_point = self.findsubarray(w=self.weights[nun_index], k=subarray_length)

            # Create the counterfactual by swapping the original instance's values for the NUN's.
            counterfactual = np.concatenate((self.instances_to_explain[instance_index][:starting_point],
                                             (self.reference_data[nun_index][starting_point:subarray_length + starting_point]),
                                             self.instances_to_explain[instance_index][subarray_length + starting_point:]))

            # Feed new instance to model and check if the probability target changed.
            counterfactual_reshaped = counterfactual.reshape(1, counterfactual.shape[0], counterfactual.shape[1])
            prob_target = model.predict(counterfactual_reshaped,verbose=0)[0][self.reference_labels[nun_index]]

        ce_label = np.argmax(model.predict(counterfactual.reshape(1, counterfactual.shape[0], counterfactual.shape[1]),
                                           verbose=0), axis=1)

        if verbose:
            print(f'Naive stage finished for instance {instance_index}')

        counterfactual_dict = {'Solution': [counterfactual], 'Window': subarray_length,
                               'Test Instance': instance_index, 'NUN Instance': nun_index, 'CE Label': ce_label[0],}
        counterfactual_df = pd.DataFrame(counterfactual_dict)
        return counterfactual_df

    def optimization(self,
                     instance_index: int,
                     nun_index: int,
                     subsequence_length: int,
                     model,
                     alpha: float = 0.5,
                     theta: float = 0.51,
                     n_partitions: int = 12,
                     population_size: int = 100,
                     maximum_number_of_generations: float = 100,
                     crossover_probability: float = 1.0,
                     mutation_probability: float = 0.9,
                     verbose: bool = False) -> pd.DataFrame:
        """
        Perform counterfactual optimization for a single instance.

        This method generates a counterfactual explanation by optimizing a subset of time steps initially
        determined by the given subsequence length. It uses the instance to be explained, its nearest
        unlike neighbor (NUN), and a trained model to guide the optimization. The process balances
        confidence and sparsity using the weighted parameter alpha, and it ensures that the counterfactual has
        a predicted probability above the threshold theta for the target class.

        Parameters:
        ----------
        ``instance_index`` : int
            Index of the instance to be explained in the sample set.
        ``nun_index`` : int
            Index of the nearest unlike neighbor (NUN) for the instance in the reference set.
        ``subsequence_length`` : int
            Initial length of the temporal subsequence to be optimized (i.e., perturbed).
        ``model`` : object
            A trained classification model with a `predict` method.
        ``alpha`` : float, default=0.5
            Weighting parameter between confidence and sparsity in the objective function.
            The higher the value, the more weight is given to confidence.
        ``theta`` : float, default=0.51
            Confidence threshold for the target class predicted probability (predicted probability ≥ theta).
        ``n_partitions`` : int, default=12
            Number of partitions for the NSGA-III reference directions.
        ``population_size`` : int, default=100
            Size of the population used in the evolutionary search algorithm.
        ``maximum_number_of_generations`` : float, default=100
            Maximum number of generations used in the evolutionary search algorithm.
        ``crossover_probability`` : float, default=1.0
            Probability of applying crossover during reproduction.
        ``mutation_probability`` : float, default=0.9
            Probability of applying mutation during reproduction.
        ``verbose`` : bool, default=False
            If True, print intermediate progress and diagnostic information.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the optimized counterfactual solutions, including the solution array,
            window size, test instance index, NUN instance index, and the predicted class label of the counterfactual.
        """

        # Initialize Values
        query = self.instances_to_explain[instance_index]

        nun = copy.deepcopy(self.reference_data[nun_index])
        solutions = pd.DataFrame(columns=["Solution", "Window", "Test Instance", "NUN Instance", 'CE Label'])

        # Start Optimization Search
        high = subsequence_length
        low = 1
        if verbose:
            print(f"Optimization of CE for Instance {instance_index} started.")
        while low <= high:
            start_time = time.time()
            window = (low + high) // 2
            if verbose:
                print(f"Optimization of CE for Instance {instance_index} in Window {window}")
            # Timestep where it starts
            if self.weights is None: #Only happens if ablation_study is True
                starting_point = 0
            else:
                starting_point = self.findsubarray((self.weights[nun_index]), window)
            end_point = starting_point + window

            # Define the Counterfactual Problem
            problem = CounterfactualProblem(query, nun, nun_index, starting_point, window, model,
                                            self.reference_labels, alpha, theta)

            # create the reference directions to be used for the optimization in NSGA3
            ref_dirs = get_reference_directions("das-dennis",
                                                n_dim=2,
                                                n_partitions=n_partitions)

            # NSGA-III Algorithm
            algorithm = NSGA3(pop_size=population_size,
                              ref_dirs=ref_dirs,
                              sampling=BinaryRandomSampling(),
                              crossover=TwoPointCrossover(prob=crossover_probability),
                              mutation=BitflipMutation(prob=mutation_probability),
                              )

            # Only do 100 generations
            termination = get_termination("n_gen", maximum_number_of_generations)

            # Run Optimization
            res = minimize(problem, algorithm, termination, seed=1, verbose=False)

            # Check if the optimization actually gave a solution
            if res.X is None:
                low = window + 1
            else:
                for x in res.X:
                    x_reshaped = res.X[0].reshape(window, query.shape[1])
                    op_counterfactual = np.copy(query)
                    op_counterfactual[starting_point:end_point][x_reshaped] = nun[starting_point:end_point][x_reshaped]

                    op_counterfactual_reshaped = op_counterfactual.reshape(1, op_counterfactual.shape[0],
                                                                           op_counterfactual.shape[1])
                    ce_label = np.argmax(model.predict(op_counterfactual_reshaped, verbose=False), axis=1)
                    row_dict = {'Solution': [op_counterfactual], 'Window': window, 'Test Instance': instance_index,
                                'NUN Instance': nun_index, 'CE Label': ce_label[0]}
                    row_df = pd.DataFrame(row_dict)
                    solutions = pd.concat([solutions, row_df], ignore_index=True)

                high = window - 1
            final_time = time.time() - start_time
            if verbose:
                print(f'Instance: {instance_index} | Window: {window} | Time: {final_time}')
        if verbose:
            print(f"Optimization of CE for Instance {instance_index} finished.")
        return solutions

    def one_pass(self, test_instance: int,
                 alpha: float = 0.5,
                 theta: float = 0.51,
                 n_partitions: int = 12,
                 population_size: int = 100,
                 maximum_number_of_generations: float = 100,
                 crossover_probability: float = 1.0,
                 mutation_probability: float = 0.9,
                 verbose: bool = False,
                 ablation_study:bool = False,
                 ablation_ts_length: int = 0):
        """
        Execute one full counterfactual generation cycle for a single test instance.

        This method serves as the base of CONFETTI's parallelization. It runs three main steps:
        1. Loads the trained model.
        2. Finds the Nearest Unlike Neighbour (NUN) of the given instance.
        3. Generates a naive counterfactual by subsequence swapping, followed by an
           optimized counterfactual using evolutionary search.

        Parameters:
        ----------
        ``test_instance`` : int
            Index of the test instance for which counterfactuals will be generated.
        ``alpha`` : float, default=0.5
            Trade-off parameter between sparsity and confidence. The higher the value,
            the more importance is given to achieving a confident prediction.
        ``theta`` : float, default=0.51
            Confidence threshold for the target class predicted probability
            (i.e., predicted probability must be ≥ theta for the counterfactual to be valid).
        ``n_partitions`` : int, default=12
            Number of partitions used to generate NSGA-III reference directions.
        ``population_size`` : int, default=100
            Size of the population in the genetic algorithm.
        ``maximum_number_of_generations`` : float, default=100
            Maximum number of generations allowed in the optimization stage.
        ``crossover_probability`` : float, default=1.0
            Probability of applying crossover during reproduction.
        ``mutation_probability`` : float, default=0.9
            Probability of applying mutation during reproduction.
        ``verbose`` : bool, default=False
            If True, prints detailed progress logs throughout execution.
        ``ablation_study`` : bool, default=False
            If True, skips the naive counterfactual and optimizes for the entire time series.
        ``ablation_ts_length`` : int, default=0
            If `ablation_study` is True, this parameter specifies the length of the time series to optimize.

        Returns:
        -------
        tuple
            - nun : int
                Index at the reference set of the nearest unlike neighbor.
            - naive : pd.DataFrame
                Counterfactual generated using the naive stage (subsequence swap).
            - optimized : pd.DataFrame
                Counterfactual refined via multi-objective optimization.
            Returns (None, None, None) if no valid NUN is found for the given instance.
        """

        # Load model
        model = keras.models.load_model(self.model_path)
        # Find the Nearest Unlike Neighbour
        nun_result = self.nearest_unlike_neighbour(query=self.instances_to_explain[test_instance],
                                            predicted_label=self.original_labels[test_instance],
                                            distance= 'euclidean',
                                            n_neighbors=1,
                                            theta=theta)
        if nun_result[0] is None:
            if verbose:
                warnings.warn(f"No NUN found for instance {test_instance} with theta {theta}.", UserWarning)
            return None, None, None
        else:
            nun = int(nun_result[1][0])

        if ablation_study:
            # If ablation study is enabled, skip the naive stage and directly optimize the entire time series
            if verbose:
                print(f'Ablation study enabled. Skipping naive stage.')
                print(f'Optimization stage started for instance {test_instance}')
                optimized = self.optimization(instance_index=test_instance,
                                              nun_index=nun,
                                              subsequence_length=ablation_ts_length,
                                              model=model,
                                              alpha=alpha,
                                              theta=theta,
                                              n_partitions=n_partitions,
                                              population_size=population_size,
                                              maximum_number_of_generations=maximum_number_of_generations,
                                              crossover_probability=crossover_probability,
                                              mutation_probability=mutation_probability,
                                              verbose=verbose)
                naive = None
                return nun, naive, optimized
        else:
            # Naive Stage
            naive = self.naive_stage(instance_index=test_instance,
                                        nun_index=nun,
                                        model=model,
                                        theta=theta,
                                        verbose=verbose)
            if verbose:
                print(f'Naive stage finished for instance {test_instance}')

                print(f'Optimization stage started for instance {test_instance}')

            # Optimization
            optimized = self.optimization(instance_index=test_instance,
                                          nun_index=nun,
                                          subsequence_length=naive.iloc[0]["Window"],
                                          model=model,
                                          alpha=alpha,
                                          theta=theta,
                                          n_partitions=n_partitions,
                                          population_size=population_size,
                                          maximum_number_of_generations=maximum_number_of_generations,
                                          crossover_probability=crossover_probability,
                                          mutation_probability=mutation_probability,
                                          verbose=verbose)

            return nun, naive, optimized


        #return nun, naive, optimized

    def parallelized_counterfactual_generator(self,
                                              instances_to_explain: np.array,
                                              reference_data: np.array,
                                              reference_weights: Optional[np.array] = None,
                                              alpha: float = 0.5,
                                              theta: float = 0.51,
                                              n_partitions: int = 12,
                                              population_size: int = 100,
                                              maximum_number_of_generations: int = 100,
                                              crossover_probability: float = 1.0,
                                              mutation_probability: float = 0.9,
                                              processes: int = 8,
                                              save_counterfactuals: bool = False,
                                              output_path: Optional[str] = None,
                                              verbose: bool = False,
                                              ablation_study: bool = False):
        """
        Generate counterfactual explanations in parallel for a set of input instances using CONFETTI.

        This method distributes the generation process across multiple processes to speed up
        the computation of counterfactuals for a batch of instances. Each instance is optimized
        independently using a genetic algorithm configured by the given parameters.

        Parameters:
        ----------
        ``instances_to_explain`` : np.array
            A NumPy array of instances for which counterfactuals should be generated. The array should be shaped
            (number of instances, timesteps, dimensions).
        ``reference_data`` : np.array
            A NumPy array of instances used as the reference set for finding nearest unlike neighbors (NUNs).
            (e.g. the training set of the model). The array should be shaped (number of instances, timesteps, dimensions).
        ``reference_weights`` : Optional[np.array]
            A NumPy array containing the class activation map (CAM) weights for each instance in the reference set.
            Each entry highlights the importance of individual time steps in the model’s class prediction.
            Shape: (n_instances, timesteps)
            Note: It is Optional only if ablation_study is True, otherwise it is required.
        ``alpha`` : float, default=0.5
            Trade-off parameter between sparsity and confidence. The higher the value,
            the more importance is given to achieving a confident prediction.
        ``theta`` : float, default=0.51
            Confidence threshold for selecting valid counterfactuals
            (i.e., predicted class probability must be ≥ theta).
        ``n_partitions`` : int, default=12
            Number of partitions for the NSGA-III reference directions.
        ``population_size`` : int, default=100
            Size of the population used in the evolutionary search algorithm.
        ``maximum_number_of_generations`` : int, default=100
            Maximum number of generations used in the evolutionary search algorithm.
        ``crossover_probability`` : float, default=1.0
            Probability of applying crossover during reproduction.
        ``mutation_probability`` : float, default=0.9
            Probability of applying mutation during reproduction.
        ``processes`` : int, default=8
            Number of parallel processes to spawn.
        ``save_counterfactuals`` : bool, default=False
            Whether to save the generated counterfactuals to disk.
        ``output_directory`` : Optional[str], optional
            Directory path to save the counterfactuals. If None, a default path is used.
        ``verbose`` : bool, default=False
            If True, print progress and debug information during execution.
        ``ablation_study`` : bool, default=False
            If True, skips the naive counterfactual and optimizes for the entire time series.

        Returns:
        -------
        pd.DataFrame, pd.DataFrame
            Two DataFrames containing the naive and optimized counterfactuals, respectively.
                Note: If ablation_study is True, the naive counterfactuals DataFrame will be empty.
            Each DataFrame includes columns for the solution array, window size, sample instance index,
            NUN instance index from reference set , and the predicted class label of the counterfactual.
        """
        self.__validate_types(locals(), context="parallelized_counterfactual_generator")
        self.instances_to_explain = instances_to_explain
        self.original_labels = np.argmax(self.model.predict(self.instances_to_explain, verbose=0), axis=1)
        self.reference_data = reference_data
        self.reference_labels = np.argmax(self.model.predict(self.reference_data, verbose=0), axis=1)
        self.weights = reference_weights

        self.naive_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Test Instance", "NUN Instance"])
        self.optimized_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Test Instance", "NUN Instance"])

        pool = Pool(processes=processes)
        wrapped_one_pass = partial(self.one_pass,
                                   alpha=alpha,
                                   theta=theta,
                                   n_partitions=n_partitions,
                                   population_size=population_size,
                                   maximum_number_of_generations=maximum_number_of_generations,
                                   crossover_probability=crossover_probability,
                                   mutation_probability=mutation_probability,
                                   verbose=verbose,
                                   ablation_study=ablation_study,
                                   ablation_ts_length=instances_to_explain.shape[1])


        res = pool.map(wrapped_one_pass, range(len(self.instances_to_explain)))
        pool.close()
        pool.join()

        for r in res:
            nun, naive_df, optimized_df = r
            self.nuns.append(nun)
            if naive_df is not None:
                self.naive_counterfactuals = pd.concat([self.naive_counterfactuals, naive_df], ignore_index=True)
            if optimized_df is not None:
                self.optimized_counterfactuals = pd.concat([self.optimized_counterfactuals, optimized_df],
                                                           ignore_index=True)

        self.optimized_counterfactuals = self.optimized_counterfactuals.groupby('Test Instance', as_index=False).apply(
            lambda x: x.drop_duplicates(subset='Window')).reset_index(drop=True)

        if save_counterfactuals:
            # If directory is not specified, use the current one
            output_directory = Path(output_path) if output_path else Path.cwd()
            output_directory.mkdir(parents=True, exist_ok=True)

            # Convert Solution arrays to space-separated strings for CSV compatibility
            if naive_df is not None:
                self.naive_counterfactuals['Solution'] = self.naive_counterfactuals['Solution'].apply(array_to_string)
            self.optimized_counterfactuals['Solution'] = self.optimized_counterfactuals['Solution'].apply(array_to_string)

            # Save files as csv
            if naive_df is not None:
                self.naive_counterfactuals.to_csv(output_directory / f'confetti_naive_counterfactuals.csv', index=False)
            self.optimized_counterfactuals.to_csv(output_directory / 'confetti_optimized_counterfactuals.csv',
                                                  index=False)
            print(f'Counterfactuals saved on {output_directory}')

        return self.naive_counterfactuals, self.optimized_counterfactuals

    def counterfactual_generator(self, instances_to_explain: np.array,
                                 reference_data: np.array,
                                 reference_weights: np.array,
                                 alpha: float = 0.5,
                                 theta: float = 0.51,
                                 n_partitions: int = 12,
                                 population_size: int = 100,
                                 maximum_number_of_generations: int = 100,
                                 crossover_probability: float = 1.0,
                                 mutation_probability: float = 0.9,
                                 save_counterfactuals: bool = False,
                                 output_path: Optional[str] = None,
                                 verbose: bool = False):
        """
        Generate counterfactual explanations sequentially for a set of input instances using CONFETTI.

        This function iterates over each instance in the input and generates counterfactual explanations
        by first applying a naive subsequence swap, followed by an optimization step based on a
        multi-objective genetic algorithm. Unlike its parallel counterpart, this version processes
        instances one at a time on a single process.

        Parameters:
        ----------
        ``instances_to_explain`` : np.array
            A NumPy array of instances for which counterfactuals should be generated.
        ``reference_data`` : np.array
            A NumPy array of instances used as the reference set for finding nearest unlike neighbors (NUNs).
        ``reference_weights`` : np.array
            A NumPy array containing the class activation map (CAM) weights for each instance in the reference set.
            Each entry highlights the importance of individual time steps in the model’s class prediction.
        ``theta`` : float, default=0.51
            Confidence threshold for selecting valid counterfactuals
            (i.e., predicted class probability must be ≥ theta).
        ``alpha`` : float, default=0.5
            Trade-off parameter between sparsity and confidence. The higher the value,
            the more importance is given to achieving a confident prediction.
        ``n_partitions`` : int, default=12
            Number of partitions used to generate NSGA-III reference directions.
        ``population_size`` : int, default=100
            Size of the population in the genetic algorithm.
        ``maximum_number_of_generations`` : int, default=100
            Maximum number of generations for the optimization algorithm.
        ``crossover_probability`` : float, default=1.0
            Probability of applying crossover during reproduction.
        ``mutation_probability`` : float, default=0.9
            Probability of applying mutation during reproduction.
        ``save_counterfactuals`` : bool, default=False
            Whether to save the generated counterfactuals to disk.
        ``output_path`` : Optional[str], optional
            Directory path to save counterfactuals. Required if `save_counterfactuals=True`.
        ``verbose`` : bool, default=False
            If True, print progress and debug information during execution.

        Returns:
        -------
        None
            Results are optionally saved to disk and/or stored internally for further use.
        """
        self.__validate_types(locals(), context= "counterfactual_generator")
        self.instances_to_explain = instances_to_explain
        self.original_labels = np.argmax(self.model.predict(self.instances_to_explain, verbose=0), axis=1)
        self.reference_data = reference_data
        self.reference_labels = np.argmax(self.model.predict(self.reference_data, verbose=0), axis=1)
        self.weights = reference_weights

        valid_instances = []
        # Find NUNs (may return None)
        for instance in range(len(instances_to_explain)):
            result = self.nearest_unlike_neighbour(query=self.instances_to_explain[instance],
                                                   predicted_label=self.original_labels[instance],
                                                   distance='euclidean',
                                                   n_neighbors=1,
                                                   theta=theta)
            if result is not None and result[1] is not None and len(result[1]) > 0:
                self.nuns.append(int(result[1][0]))
                valid_instances.append(instance)
            else:
                print(f"Skipping instance {instance}: No valid NUN found.")

        self.nuns = np.array(self.nuns)
        test_instances = np.array(valid_instances)

            # Naive Counterfactuals
        self.naive_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Test Instance", "NUN Instance"])
        for test_instance, nun in zip(test_instances, self.nuns):
            naive_df = self.naive_stage(instance_index=test_instance,
                                           nun_index=nun,
                                           model=self.model,
                                           theta=theta)
            if naive_df is not None:
                self.naive_counterfactuals = pd.concat([self.naive_counterfactuals, naive_df], ignore_index=True)


        self.optimized_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Test Instance", "NUN Instance"])

        for test_instance in test_instances:
            # Filter naive counterfactuals for this test instance
            naive_row = self.naive_counterfactuals[
                self.naive_counterfactuals["Test Instance"] == test_instance]

            if naive_row.empty:
                print(f"Skipping optimization for instance {test_instance}: no naive counterfactual found.")
                continue  # No naive CE means we cannot optimize

            window_val = naive_row["Window"].iloc[0]
            nun = self.nuns[test_instances.tolist().index(test_instance)]

            opt_df = self.optimization(instance_index=test_instance,
                                        nun_index=nun,
                                        subsequence_length=window_val,
                                        model=self.model,
                                        alpha=alpha,
                                        theta=theta,
                                        n_partitions=n_partitions,
                                        population_size=population_size,
                                        maximum_number_of_generations=maximum_number_of_generations,
                                        crossover_probability=crossover_probability,
                                        mutation_probability=mutation_probability,
                                        verbose=verbose)
            if opt_df is not None:
                self.optimized_counterfactuals = pd.concat(
                    [self.optimized_counterfactuals, opt_df], ignore_index=True
                )

        self.optimized_counterfactuals = self.optimized_counterfactuals.groupby('Test Instance',
                                                                                as_index=False).apply(
            lambda x: x.drop_duplicates(subset='Window')
        ).reset_index(drop=True)

        if save_counterfactuals:
            # Create the directory if it doesn't exist
            output_directory = Path(output_path) if output_path else Path.cwd()
            output_directory.mkdir(parents=True, exist_ok=True)

            # Convert Solution arrays to space-separated strings for CSV compatibility
            self.naive_counterfactuals['Solution'] = self.naive_counterfactuals['Solution'].apply(array_to_string)

            # Save the naive counterfactuals
            self.naive_counterfactuals.to_csv(output_directory / 'confetti_naive_counterfactuals.csv', index=False)

            self.optimized_counterfactuals['Solution'] = self.optimized_counterfactuals['Solution'].apply(
                array_to_string)
            self.optimized_counterfactuals.to_csv(output_directory / 'confetti_optimized_counterfactuals.csv',
                                                  index=False)

    @staticmethod
    def __validate_types(arguments: dict, context: str = ""):
        """
        Validates argument types for CONFETTI generators. Handles both parallel and non-parallel modes.
        """
        # Shared arguments
        expected_types = {
            "instances_to_explain": np.ndarray,
            "reference_data": np.ndarray,
            "alpha": float,
            "theta": float,
            "n_partitions": int,
            "population_size": int,
            "maximum_number_of_generations": int,
            "crossover_probability": float,
            "mutation_probability": float,
            "save_counterfactuals": bool,
            "verbose": bool,
        }

        # Add only if using the parallel version
        if context == "parallelized_counterfactual_generator":
            expected_types["processes"] = int

        for name, expected in expected_types.items():
            value = arguments.get(name)
            if not isinstance(value, expected):
                raise CONFETTIConfigurationError(
                    f"{context + ': ' if context else ''}"
                    f"Parameter `{name}` must be of type {expected.__name__}, "
                    f"but got {type(value).__name__}."
                )

        # Special case: reference_weights can be None only if ablation_study is True
        reference_weights = arguments.get("reference_weights")
        ablation_study = arguments.get("ablation_study", False)
        if reference_weights is None:
            if not ablation_study:
                raise CONFETTIConfigurationError(
                    f"{context + ': ' if context else ''}"
                    "`reference_weights` must not be None unless `ablation_study=True`."
                )
        elif not isinstance(reference_weights, np.ndarray):
            raise CONFETTIConfigurationError(
                f"{context + ': ' if context else ''}"
                f"Parameter `reference_weights` must be of type np.ndarray, but got {type(reference_weights).__name__}."
            )

        output_path = arguments.get("output_path")
        save = arguments.get("save_counterfactuals")
        if save:
            if output_path is None or not isinstance(output_path, str):
                raise CONFETTIConfigurationError(
                    f"{context + ': ' if context else ''}"
                    "`output_path` must be a non-None string when `save_counterfactuals=True`."
                )
