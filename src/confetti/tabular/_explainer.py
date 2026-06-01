from __future__ import annotations

import time
from functools import partial
from multiprocessing import Pool
from collections.abc import Callable
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from confetti.algorithm import NSGA3, das_dennis, minimize
from confetti.algorithm.crossover import TwoPointCrossover
from confetti.algorithm.mutation import BitflipMutation
from confetti.algorithm.sampling import BinaryRandomSampling
from confetti.constraints import RelationConstraint
from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError
from confetti.structs import Counterfactual, CounterfactualSet, CounterfactualResults
from confetti.tabular._tabular_problem import TabularCounterfactualProblem, _SUPPORTED_METRICS


class _TabularPredictorAdapter:
    """Normalize ``predict_proba`` / ``predict`` to a single ``predict`` method.

    When a ``preprocessor`` is provided, it is applied to the input
    before every call to the underlying model.

    Parameters
    ----------
    ``model`` : object
        A fitted classifier with ``predict_proba`` or ``predict``.
    ``preprocessor`` : Callable or None, default=None
        Optional transform applied to ``X`` before prediction.
        Receives a ``pd.DataFrame`` or ``np.ndarray`` depending on
        whether the input data contained categorical columns.
    """

    def __init__(
        self,
        model,
        preprocessor: Callable[[pd.DataFrame | np.ndarray], pd.DataFrame | np.ndarray] | None = None,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_classes)``.

        Parameters
        ----------
        ``X`` : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        transformed = self._preprocessor(X) if self._preprocessor is not None else X
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(transformed)
        return self._model.predict(transformed)


class TabularCONFETTI:
    """Generate counterfactual explanations for tabular classifiers.

    Uses NSGA-III multi-objective optimization to find minimal feature
    perturbations that change the predicted class.  For each instance,
    the algorithm searches over binary masks that decide, per feature,
    whether to keep the original value or adopt the value from the
    nearest unlike neighbour (NUN).

    Parameters
    ----------
    ``model`` : object
        A fitted classifier with a ``predict_proba`` or ``predict``
        method returning class probabilities.
    ``feature_names`` : list[str] or None, default=None
        Column names for output DataFrames.  Inferred from input
        DataFrames when not provided.
    ``preprocessor`` : Callable[[pd.DataFrame | np.ndarray], np.ndarray] or None, default=None
        Optional transform applied before every call to
        ``model.predict_proba`` (or ``model.predict``).  Use this when
        the model expects a different encoding than the raw feature
        space (e.g. one-hot encoding, standard scaling).  When the
        input data contains categorical columns, the preprocessor
        receives a ``pd.DataFrame`` with original category labels;
        otherwise it receives a ``np.ndarray``.  The GA search, NUN
        lookup, proximity, and sparsity objectives all operate on the
        ordinal-encoded features — only predictions are affected.

    Raises
    ------
    CONFETTIConfigurationError
        If *model* lacks a prediction interface, if *feature_names* is
        not a list of strings, if *feature_names* contains duplicates,
        or if *preprocessor* is not callable.
    """

    def __init__(
        self,
        model,
        feature_names: list[str] | None = None,
        preprocessor: Callable[[pd.DataFrame | np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._preprocessor = preprocessor
        self._validate_init_params()

        self._classifier = _TabularPredictorAdapter(model, preprocessor=preprocessor)
        self._categorical_mappings: Optional[dict[int, dict[int, Any]]] = None
        self._instances_np: Optional[np.ndarray] = None
        self._reference_np: Optional[np.ndarray] = None
        self._original_labels: Optional[np.ndarray] = None
        self._reference_labels: Optional[np.ndarray] = None
        self._cat_mask: Optional[np.ndarray] = None
        self._ranges: Optional[np.ndarray] = None
        self._immutable_mask: Optional[np.ndarray] = None
        self._relation_constraints: Optional[list[RelationConstraint]] = None

    @property
    def model(self):
        """Return the underlying classification model."""
        return self._model

    @property
    def feature_names(self) -> list[str] | None:
        """Return the feature names used for output DataFrames."""
        return self._feature_names

    def _require_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return internal arrays or raise if not yet initialized.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(instances_np, reference_np, original_labels, reference_labels)``.

        Raises
        ------
        CONFETTIConfigurationError
            If :meth:`generate_counterfactuals` has not been called yet.
        """
        if (
            self._instances_np is None
            or self._reference_np is None
            or self._original_labels is None
            or self._reference_labels is None
        ):
            raise CONFETTIConfigurationError(
                message="Internal state not initialized. Call generate_counterfactuals() first.",
                source="TabularCONFETTI._require_state",
            )
        return self._instances_np, self._reference_np, self._original_labels, self._reference_labels

    def generate_counterfactuals(
        self,
        instances_to_explain: pd.DataFrame | np.ndarray,
        reference_data: pd.DataFrame | np.ndarray,
        alpha: float = 0.5,
        theta: float = 0.51,
        n_partitions: int = 3,
        population_size: int = 100,
        maximum_number_of_generations: int = 100,
        crossover_probability: float = 1.0,
        mutation_probability: float = 0.9,
        optimize_confidence: bool = True,
        optimize_sparsity: bool = True,
        optimize_proximity: bool = False,
        proximity_distance: str = "euclidean",
        categorical_features: list[int] | None = None,
        immutable_features: list[str] | list[int] | None = None,
        relation_constraints: list[RelationConstraint] | None = None,
        processes: int | None = None,
        verbose: bool = False,
    ) -> CounterfactualResults | None:
        """Generate counterfactual explanations for one or more instances.

        Parameters
        ----------
        ``instances_to_explain`` : pd.DataFrame or np.ndarray
            Instances to explain, shape ``(n_instances, n_features)``.
        ``reference_data`` : pd.DataFrame or np.ndarray
            Reference dataset for NUN search, shape
            ``(n_reference, n_features)``.
        ``alpha`` : float, default=0.5
            Trade-off weight between confidence and sparsity when
            selecting the best counterfactual.
        ``theta`` : float, default=0.51
            Minimum predicted probability for the target class.
        ``n_partitions`` : int, default=3
            Number of partitions for NSGA-III reference directions.
        ``population_size`` : int, default=100
            Population size for the evolutionary algorithm.
        ``maximum_number_of_generations`` : int, default=100
            Maximum generations for the search.
        ``crossover_probability`` : float, default=1.0
            Probability of crossover.
        ``mutation_probability`` : float, default=0.9
            Probability of mutation per bit.
        ``optimize_confidence`` : bool, default=True
            Include confidence maximization as an objective.
        ``optimize_sparsity`` : bool, default=True
            Include sparsity minimization as an objective.
        ``optimize_proximity`` : bool, default=False
            Include proximity minimization as an objective.
        ``proximity_distance`` : str, default="euclidean"
            Distance metric for proximity (``"euclidean"``,
            ``"gower"``, or ``"manhattan"``).
        ``categorical_features`` : list[int] or None, default=None
            Indices of categorical features, used by the ``"gower"``
            proximity metric.  When ``proximity_distance="gower"``:

            - If provided, these indices mark which features are
              categorical (binary match/mismatch).  All other features
              are treated as numerical (range-normalized absolute
              difference).
            - If ``None`` and the input data is a DataFrame with
              non-numeric columns, categorical features are
              auto-detected from the DataFrame schema.
            - If ``None`` and the input data is a numeric array or
              numeric DataFrame, all features are treated as
              numerical and Gower reduces to range-normalized
              Manhattan distance.

            Ignored when *proximity_distance* is not ``"gower"``.
        ``immutable_features`` : list[str] or list[int] or None, default=None
            Features that must not be modified in counterfactual
            explanations.  Accepts column names (``list[str]``) when
            the input is a DataFrame or *feature_names* were provided,
            or column indices (``list[int]``) for any input type.
            Immutable features are guaranteed to retain their original
            values and are excluded from the sparsity objective.
        ``relation_constraints`` : list[RelationConstraint] or None, default=None
            Inter-feature relational constraints.  When provided,
            constraint violations are added as an additional inequality
            constraint in the GA.  ``Equal`` constraints with a single
            ``Feature`` left operand are also repaired in-place before
            objective computation.
        ``processes`` : int or None, default=None
            Worker processes for parallel execution.  Sequential if
            ``None``.
        ``verbose`` : bool, default=False
            Print progress messages.

        Returns
        -------
        CounterfactualResults or None
            All generated counterfactual sets, or ``None`` if no valid
            counterfactuals were found.

        Raises
        ------
        CONFETTIConfigurationError
            If parameter values are invalid.
        CONFETTIDataTypeError
            If input types are wrong.
        """
        self._validate_counterfactual_params(
            instances_to_explain=instances_to_explain,
            reference_data=reference_data,
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
            categorical_features=categorical_features,
            relation_constraints=relation_constraints,
            processes=processes,
        )

        self._instances_np, self._reference_np = self._prepare_data(
            instances_to_explain,
            reference_data,
        )

        self._immutable_mask = self._resolve_immutable_features(
            immutable_features,
            instances_to_explain,
            self._instances_np.shape[1],
        )

        self._cat_mask, self._ranges = self._build_gower_params(
            proximity_distance,
            categorical_features,
            self._reference_np,
        )
        self._classifier = self._build_classifier()

        self._original_labels = np.argmax(
            self._classifier.predict(self._instances_np),
            axis=1,
        )
        self._reference_labels = np.argmax(
            self._classifier.predict(self._reference_np),
            axis=1,
        )

        self._relation_constraints = relation_constraints

        if processes is not None:
            return self._parallelized_generator(
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
                processes=processes,
                verbose=verbose,
            )

        return self._generator(
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
            verbose=verbose,
        )

    def _prepare_data(
        self,
        instances: pd.DataFrame | np.ndarray,
        reference: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to float64 ndarrays, ordinal-encoding categoricals.

        When either input is a DataFrame with non-numeric columns, those
        columns are ordinal-encoded and the reverse mapping is stored in
        ``self._categorical_mappings`` so that arrays can later be
        decoded back to original labels.

        Parameters
        ----------
        ``instances`` : pd.DataFrame or np.ndarray
            Instances to explain.
        ``reference`` : pd.DataFrame or np.ndarray
            Reference dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(instances_np, reference_np)`` as float64 arrays.
        """
        if self._feature_names is None:
            if isinstance(instances, pd.DataFrame):
                self._feature_names = list(instances.columns)
            elif isinstance(reference, pd.DataFrame):
                self._feature_names = list(reference.columns)

        self._categorical_mappings = None

        ref_df = reference if isinstance(reference, pd.DataFrame) else None
        inst_df = instances if isinstance(instances, pd.DataFrame) else None
        schema_df = ref_df if ref_df is not None else inst_df

        if schema_df is not None:
            cat_mappings: dict[int, dict[int, Any]] = {}
            for i, col in enumerate(schema_df.columns):
                if not pd.api.types.is_numeric_dtype(schema_df[col]):
                    unique_values = sorted(schema_df[col].dropna().unique(), key=str)
                    cat_mappings[i] = {idx: val for idx, val in enumerate(unique_values)}
            if cat_mappings:
                self._categorical_mappings = cat_mappings

        reference_np = self._encode_dataframe(ref_df) if ref_df is not None else np.asarray(reference, dtype=np.float64)
        instances_np = (
            self._encode_dataframe(inst_df) if inst_df is not None else np.asarray(instances, dtype=np.float64)
        )

        if self._feature_names is None:
            self._feature_names = [f"feature_{i}" for i in range(instances_np.shape[1])]

        return instances_np, reference_np

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Ordinal-encode categorical columns and convert to float64.

        Parameters
        ----------
        ``df`` : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(n_rows, n_cols)``.
        """
        if self._categorical_mappings is None:
            return df.to_numpy(dtype=np.float64)

        result = np.empty((len(df), df.shape[1]), dtype=np.float64)
        for i, col in enumerate(df.columns):
            if i in self._categorical_mappings:
                int_to_value = self._categorical_mappings[i]
                value_to_int = {v: k for k, v in int_to_value.items()}
                result[:, i] = df[col].map(value_to_int).to_numpy(dtype=np.float64)
            else:
                result[:, i] = df[col].to_numpy(dtype=np.float64)
        return result

    def _decode_to_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """Convert an ordinal-encoded array back to a DataFrame with original labels.

        Parameters
        ----------
        ``X`` : np.ndarray
            Array of shape ``(n_rows, n_features)``.

        Returns
        -------
        pd.DataFrame
            DataFrame with original category labels restored.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        feature_names = self._feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        data: dict[str, Any] = {}
        for i, name in enumerate(feature_names):
            if self._categorical_mappings is not None and i in self._categorical_mappings:
                mapping = self._categorical_mappings[i]
                data[name] = [mapping.get(int(round(v)), v) for v in X[:, i]]
            else:
                data[name] = X[:, i]
        return pd.DataFrame(data)

    def _build_classifier(self) -> _TabularPredictorAdapter:
        """Build the predictor adapter, wrapping the preprocessor with decoding if needed.

        Returns
        -------
        _TabularPredictorAdapter
            Adapter that decodes ordinal-encoded arrays back to
            original-space DataFrames before calling the preprocessor
            and model.
        """
        if self._categorical_mappings is not None:
            user_preprocessor = self._preprocessor

            def _decoding_preprocessor(X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
                df = self._decode_to_dataframe(np.asarray(X))
                if user_preprocessor is not None:
                    return user_preprocessor(df)
                return df

            return _TabularPredictorAdapter(self._model, preprocessor=_decoding_preprocessor)

        return _TabularPredictorAdapter(self._model, preprocessor=self._preprocessor)

    def _build_gower_params(
        self,
        proximity_distance: str,
        categorical_features: list[int] | None,
        reference_np: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute ``cat_mask`` and ``ranges`` for Gower distance.

        Parameters
        ----------
        ``proximity_distance`` : str
            The proximity metric name.
        ``categorical_features`` : list[int] or None
            User-provided categorical feature indices.
        ``reference_np`` : np.ndarray
            Reference data of shape ``(n_samples, n_features)``.

        Returns
        -------
        tuple[np.ndarray or None, np.ndarray or None]
            ``(cat_mask, ranges)`` when *proximity_distance* is
            ``"gower"``, otherwise ``(None, None)``.
        """
        if proximity_distance != "gower":
            return None, None

        n_features = reference_np.shape[1]
        cat_mask = np.zeros(n_features, dtype=np.bool_)

        if categorical_features is not None:
            cat_mask[categorical_features] = True
        elif self._categorical_mappings is not None:
            for idx in self._categorical_mappings:
                cat_mask[idx] = True

        ranges = np.ptp(reference_np, axis=0).astype(np.float64)
        ranges[cat_mask] = 0.0

        return cat_mask, ranges

    def _resolve_immutable_features(
        self,
        immutable_features: list[str] | list[int] | None,
        instances_to_explain: pd.DataFrame | np.ndarray,
        n_features: int,
    ) -> np.ndarray | None:
        """Resolve *immutable_features* to a boolean mask.

        Parameters
        ----------
        ``immutable_features`` : list[str] or list[int] or None
            User-provided immutable feature specification.
        ``instances_to_explain`` : pd.DataFrame or np.ndarray
            Original input data (used to determine whether column
            names are available).
        ``n_features`` : int
            Total number of features.

        Returns
        -------
        np.ndarray or None
            Boolean mask of shape ``(n_features,)`` where ``True``
            marks immutable features, or ``None`` when no features
            are immutable.

        Raises
        ------
        CONFETTIDataTypeError
            If *immutable_features* contains mixed types.
        CONFETTIConfigurationError
            If column names cannot be resolved, indices are out of
            range, duplicates are present, or all features are
            immutable.
        """
        if immutable_features is None or len(immutable_features) == 0:
            return None

        all_str = all(isinstance(f, str) for f in immutable_features)
        all_int = all(isinstance(f, int) for f in immutable_features)

        if not all_str and not all_int:
            raise CONFETTIDataTypeError(
                message="immutable_features must be all strings or all integers, not a mix.",
                param="immutable_features",
                hint="Use column names (['age', 'gender']) or indices ([0, 2]), not both.",
            )

        if all_str:
            str_features: list[str] = [str(f) for f in immutable_features]
            has_real_names = (
                isinstance(instances_to_explain, pd.DataFrame)
                or self._feature_names != [f"feature_{i}" for i in range(n_features)]
            )
            if not has_real_names:
                raise CONFETTIConfigurationError(
                    message="String immutable_features require DataFrame input or explicit feature_names.",
                    param="immutable_features",
                    hint="Pass a DataFrame, set feature_names at init, or use integer indices.",
                )
            if len(str_features) != len(set(str_features)):
                raise CONFETTIConfigurationError(
                    message="immutable_features contains duplicate names.",
                    param="immutable_features",
                )
            assert self._feature_names is not None
            indices: list[int] = []
            for name in str_features:
                if name not in self._feature_names:
                    raise CONFETTIConfigurationError(
                        message=f"immutable feature '{name}' not found in feature names.",
                        param="immutable_features",
                        hint=f"Available features: {self._feature_names}.",
                    )
                indices.append(self._feature_names.index(name))
        else:
            int_features: list[int] = [int(f) for f in immutable_features]
            if len(int_features) != len(set(int_features)):
                raise CONFETTIConfigurationError(
                    message="immutable_features contains duplicate indices.",
                    param="immutable_features",
                )
            for idx in int_features:
                if idx < 0 or idx >= n_features:
                    raise CONFETTIConfigurationError(
                        message=f"immutable_features index {idx} is out of range for {n_features} features.",
                        param="immutable_features",
                        hint=f"Indices must be in [0, {n_features - 1}].",
                    )
            indices = int_features

        if len(indices) >= n_features:
            raise CONFETTIConfigurationError(
                message="All features are immutable — no counterfactual search is possible.",
                param="immutable_features",
                hint="Leave at least one feature mutable.",
            )

        mask = np.zeros(n_features, dtype=np.bool_)
        mask[indices] = True
        return mask

    def _nearest_unlike_neighbour(
        self,
        query: np.ndarray,
        predicted_label: int,
        distance: str = "euclidean",
        theta: float = 0.51,
    ) -> int | None:
        """Find the nearest reference instance with a different predicted label.

        Parameters
        ----------
        ``query`` : np.ndarray
            Feature vector of shape ``(n_features,)``.
        ``predicted_label`` : int
            Predicted class of the query.
        ``distance`` : str, default="euclidean"
            Distance metric for neighbour search.
        ``theta`` : float, default=0.51
            Minimum confidence for a neighbour to be valid.

        Returns
        -------
        int or None
            Index into the reference dataset, or ``None`` if no valid
            NUN exists.
        """
        _, reference_np, _, reference_labels = self._require_state()

        unlike_mask = reference_labels != predicted_label
        unlike_indices = np.where(unlike_mask)[0]

        if len(unlike_indices) == 0:
            return None

        unlike_data = reference_np[unlike_indices]

        nn_metric = "euclidean" if distance == "gower" else distance
        nn = NearestNeighbors(n_neighbors=min(len(unlike_indices), 5), metric=nn_metric)
        nn.fit(unlike_data)
        _, ind = nn.kneighbors(query.reshape(1, -1))

        for local_idx in ind[0]:
            actual_idx = unlike_indices[local_idx]
            probs = self._classifier.predict(reference_np[actual_idx].reshape(1, -1))
            confidence = np.max(probs, axis=1)[0]
            if confidence >= theta:
                return int(actual_idx)

        return None

    def _optimization(
        self,
        instance_index: int,
        nun_index: int,
        alpha: float,
        theta: float,
        n_partitions: int,
        population_size: int,
        maximum_number_of_generations: int,
        crossover_probability: float,
        mutation_probability: float,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        verbose: bool = False,
    ) -> CounterfactualSet | None:
        """Run NSGA-III to find optimal counterfactual masks for one instance.

        Parameters
        ----------
        ``instance_index`` : int
            Index of the instance to explain.
        ``nun_index`` : int
            Index of the NUN in the reference dataset.
        ``alpha`` : float
            Confidence-sparsity trade-off weight.
        ``theta`` : float
            Minimum confidence threshold.
        ``n_partitions`` : int
            Reference direction granularity.
        ``population_size`` : int
            EA population size.
        ``maximum_number_of_generations`` : int
            Maximum EA generations.
        ``crossover_probability`` : float
            Crossover probability.
        ``mutation_probability`` : float
            Mutation probability.
        ``optimize_confidence`` : bool
            Optimize confidence objective.
        ``optimize_sparsity`` : bool
            Optimize sparsity objective.
        ``optimize_proximity`` : bool
            Optimize proximity objective.
        ``proximity_distance`` : str
            Proximity metric.
        ``verbose`` : bool, default=False
            Print progress.

        Returns
        -------
        CounterfactualSet or None
            Generated counterfactuals, or ``None`` if no valid
            solutions were found.
        """
        instances_np, reference_np, original_labels, reference_labels = self._require_state()

        query = instances_np[instance_index]
        nun = reference_np[nun_index].copy()

        n_obj = int(optimize_confidence) + int(optimize_sparsity) + int(optimize_proximity)

        if verbose:
            start_time = time.time()
            print(f"Optimization started for instance {instance_index}.")

        problem = TabularCounterfactualProblem(
            original_instance=query,
            nun_instance=nun,
            nun_index=nun_index,
            classifier=self._classifier,
            reference_labels=reference_labels,
            optimize_confidence=optimize_confidence,
            optimize_sparsity=optimize_sparsity,
            optimize_proximity=optimize_proximity,
            proximity_distance=proximity_distance,
            theta=theta,
            cat_mask=self._cat_mask,
            ranges=self._ranges,
            immutable_mask=self._immutable_mask,
            relation_constraints=self._relation_constraints,
            feature_names=self._feature_names,
        )

        reference_directions = das_dennis(n_obj, n_partitions)

        algorithm = NSGA3(
            pop_size=population_size,
            ref_dirs=reference_directions,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=crossover_probability),
            mutation=BitflipMutation(prob=mutation_probability),
        )

        result = minimize(problem, algorithm, maximum_number_of_generations, seed=1, verbose=False)

        if result.X is None:
            if verbose:
                print(f"No valid solutions for instance {instance_index}.")
            return None

        counterfactuals_np = np.where(result.X, nun, query)
        if self._immutable_mask is not None:
            counterfactuals_np[:, self._immutable_mask] = query[self._immutable_mask]
        predictions = self._classifier.predict(counterfactuals_np)
        predicted_labels = np.argmax(predictions, axis=1)

        all_counterfactuals = self._build_counterfactual_list(counterfactuals_np, predicted_labels)

        if result.F is None:
            return None
        objective_values = result.F
        if optimize_confidence and optimize_sparsity:
            best = self._select_best_solution(all_counterfactuals, objective_values, alpha)
        else:
            best = None

        original_display = self._to_display(query)
        nun_display = self._to_display(nun)

        counterfactual_set = CounterfactualSet(
            original_instance=original_display,
            original_label=original_labels[instance_index].item(),
            nearest_unlike_neighbour=nun_display,
            best_solution=best,
            all_counterfactuals=all_counterfactuals,
        )

        if verbose:
            elapsed = time.time() - start_time
            print(f"Instance {instance_index} done in {elapsed:.2f}s.")

        return counterfactual_set

    def _build_counterfactual_list(
        self,
        counterfactuals_np: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> List[Counterfactual]:
        """Wrap numpy counterfactuals into Counterfactual objects with display values.

        Parameters
        ----------
        ``counterfactuals_np`` : np.ndarray
            Counterfactual matrix of shape ``(n_solutions, n_features)``.
        ``predicted_labels`` : np.ndarray
            Predicted class per solution, shape ``(n_solutions,)``.

        Returns
        -------
        list[Counterfactual]
            One :class:`Counterfactual` per solution.
        """
        results = []
        for cf_row, label in zip(counterfactuals_np, predicted_labels):
            display = self._to_display(cf_row)
            results.append(Counterfactual(counterfactual=display, label=int(label)))
        return results

    def _to_display(self, arr: np.ndarray) -> pd.DataFrame | np.ndarray:
        """Convert a feature vector to a display DataFrame when possible.

        Parameters
        ----------
        ``arr`` : np.ndarray
            Feature vector of shape ``(n_features,)``.

        Returns
        -------
        pd.DataFrame or np.ndarray
            A single-row DataFrame with original category labels when
            categorical mappings exist, a plain DataFrame when feature
            names are known, or the raw array otherwise.
        """
        if self._categorical_mappings is not None:
            return self._decode_to_dataframe(arr)
        if self._feature_names is not None:
            return pd.DataFrame([arr], columns=self._feature_names)
        return arr

    def _generator(
        self,
        alpha: float,
        theta: float,
        n_partitions: int,
        population_size: int,
        maximum_number_of_generations: int,
        crossover_probability: float,
        mutation_probability: float,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        verbose: bool,
    ) -> CounterfactualResults | None:
        """Generate counterfactuals sequentially for all instances.

        Parameters
        ----------
        (See :meth:`generate_counterfactuals` for parameter descriptions.)

        Returns
        -------
        CounterfactualResults or None
            All counterfactual sets, or ``None`` if none were found.
        """
        instances_np, _, original_labels, _ = self._require_state()

        counterfactual_sets: List[CounterfactualSet] = []

        for idx in range(len(instances_np)):
            nun_index = self._nearest_unlike_neighbour(
                query=instances_np[idx],
                predicted_label=int(original_labels[idx]),
                distance=proximity_distance,
                theta=theta,
            )

            if nun_index is None:
                if verbose:
                    print(f"Skipping instance {idx}: no valid NUN found.")
                continue

            result = self._optimization(
                instance_index=idx,
                nun_index=nun_index,
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
                verbose=verbose,
            )

            if result is not None:
                counterfactual_sets.append(result)

        if counterfactual_sets:
            return CounterfactualResults(counterfactual_sets=counterfactual_sets)
        return None

    def _parallelized_generator(
        self,
        alpha: float,
        theta: float,
        n_partitions: int,
        population_size: int,
        maximum_number_of_generations: int,
        crossover_probability: float,
        mutation_probability: float,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        processes: int,
        verbose: bool,
    ) -> CounterfactualResults | None:
        """Generate counterfactuals in parallel for all instances.

        Parameters
        ----------
        (See :meth:`generate_counterfactuals` for parameter descriptions.)

        Returns
        -------
        CounterfactualResults or None
            All counterfactual sets from parallel execution, or ``None``.
        """
        wrapped = partial(
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
            verbose=verbose,
        )

        instances_np, _, _, _ = self._require_state()

        pool = Pool(processes=processes)
        results = pool.map(wrapped, range(len(instances_np)))
        pool.close()
        pool.join()

        counterfactual_sets = [r for r in results if r is not None]

        if counterfactual_sets:
            return CounterfactualResults(counterfactual_sets=counterfactual_sets)
        return None

    def _one_pass(
        self,
        instance_index: int,
        alpha: float,
        theta: float,
        n_partitions: int,
        population_size: int,
        maximum_number_of_generations: int,
        crossover_probability: float,
        mutation_probability: float,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        verbose: bool,
    ) -> CounterfactualSet | None:
        """Full pipeline for a single instance (used by parallel generator).

        Parameters
        ----------
        ``instance_index`` : int
            Index of the instance to explain.
        (See :meth:`generate_counterfactuals` for other parameters.)

        Returns
        -------
        CounterfactualSet or None
            Counterfactuals for the instance, or ``None``.
        """
        instances_np, _, original_labels, _ = self._require_state()

        nun_index = self._nearest_unlike_neighbour(
            query=instances_np[instance_index],
            predicted_label=int(original_labels[instance_index]),
            distance=proximity_distance,
            theta=theta,
        )

        if nun_index is None:
            if verbose:
                print(f"Skipping instance {instance_index}: no valid NUN found.")
            return None

        return self._optimization(
            instance_index=instance_index,
            nun_index=nun_index,
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
            verbose=verbose,
        )

    @staticmethod
    def _select_best_solution(
        counterfactuals: List[Counterfactual],
        objective_values: np.ndarray,
        alpha: float,
    ) -> Counterfactual:
        """Select the best counterfactual via weighted confidence-sparsity score.

        Parameters
        ----------
        ``counterfactuals`` : list[Counterfactual]
            All candidate counterfactuals.
        ``objective_values`` : np.ndarray
            Objective matrix of shape ``(n_solutions, n_objectives)``.
        ``alpha`` : float
            Weight for confidence (``1 - alpha`` for sparsity).

        Returns
        -------
        Counterfactual
            The highest-scoring counterfactual.
        """
        scores_matrix = objective_values[:, :2].copy()
        scores_matrix[:, 0] = -scores_matrix[:, 0]
        weights = np.array([alpha, 1 - alpha])
        scores = np.dot(scores_matrix, weights)
        return counterfactuals[int(np.argmax(scores))]

    def _validate_init_params(self) -> None:
        """Validate constructor arguments.

        Raises
        ------
        CONFETTIConfigurationError
            If *model* lacks a prediction interface, *feature_names* is
            invalid, *feature_names* has duplicates, or *preprocessor*
            is not callable.
        """
        if not hasattr(self._model, "predict_proba") and not hasattr(self._model, "predict"):
            raise CONFETTIConfigurationError(
                message="Model must expose a predict_proba() or predict() method.",
                param="model",
                hint="Pass a fitted sklearn classifier or any model with predict_proba/predict.",
            )

        if self._preprocessor is not None and not callable(self._preprocessor):
            raise CONFETTIConfigurationError(
                message="preprocessor must be callable.",
                param="preprocessor",
                hint="Pass a function like sklearn's ColumnTransformer.transform or a custom lambda.",
            )

        if self._feature_names is not None:
            if not isinstance(self._feature_names, list) or not all(isinstance(n, str) for n in self._feature_names):
                raise CONFETTIConfigurationError(
                    message="feature_names must be a list of strings.",
                    param="feature_names",
                    hint="Pass a list like ['age', 'income', 'status'].",
                )
            if len(self._feature_names) != len(set(self._feature_names)):
                raise CONFETTIConfigurationError(
                    message="feature_names contains duplicates.",
                    param="feature_names",
                    hint="Ensure all feature names are unique.",
                )

    @staticmethod
    def _validate_counterfactual_params(
        instances_to_explain,
        reference_data,
        alpha: float,
        theta: float,
        n_partitions: int,
        population_size: int,
        maximum_number_of_generations: int,
        crossover_probability: float,
        mutation_probability: float,
        optimize_confidence: bool,
        optimize_sparsity: bool,
        optimize_proximity: bool,
        proximity_distance: str,
        categorical_features: list[int] | None,
        relation_constraints: list[RelationConstraint] | None = None,
        processes: int | None = None,
    ) -> None:
        """Validate ``generate_counterfactuals`` arguments.

        Raises
        ------
        CONFETTIDataTypeError
            If input types are wrong.
        CONFETTIConfigurationError
            If parameter values are invalid.
        """
        if not isinstance(instances_to_explain, (pd.DataFrame, np.ndarray)):
            raise CONFETTIDataTypeError(
                message=f"instances_to_explain must be a pd.DataFrame or np.ndarray, got {type(instances_to_explain).__name__}.",
                param="instances_to_explain",
            )
        if not isinstance(reference_data, (pd.DataFrame, np.ndarray)):
            raise CONFETTIDataTypeError(
                message=f"reference_data must be a pd.DataFrame or np.ndarray, got {type(reference_data).__name__}.",
                param="reference_data",
            )

        inst_shape = (
            instances_to_explain.shape
            if isinstance(instances_to_explain, np.ndarray)
            else instances_to_explain.values.shape
        )
        ref_shape = reference_data.shape if isinstance(reference_data, np.ndarray) else reference_data.values.shape

        if len(inst_shape) != 2:
            raise CONFETTIConfigurationError(
                message=f"instances_to_explain must be 2-D, got {len(inst_shape)}-D.",
                param="instances_to_explain",
                hint="Pass tabular data of shape (n_samples, n_features).",
            )
        if len(ref_shape) != 2:
            raise CONFETTIConfigurationError(
                message=f"reference_data must be 2-D, got {len(ref_shape)}-D.",
                param="reference_data",
                hint="Pass tabular data of shape (n_samples, n_features).",
            )
        if inst_shape[1] != ref_shape[1]:
            raise CONFETTIConfigurationError(
                message=f"Feature count mismatch: instances_to_explain has {inst_shape[1]} features, reference_data has {ref_shape[1]}.",
                param="instances_to_explain",
                hint="Both datasets must have the same number of features.",
            )

        if isinstance(instances_to_explain, pd.DataFrame) and isinstance(reference_data, pd.DataFrame):
            if list(instances_to_explain.columns) != list(reference_data.columns):
                raise CONFETTIConfigurationError(
                    message="Column names of instances_to_explain and reference_data do not match.",
                    param="instances_to_explain",
                    hint="Ensure both DataFrames have identical column names in the same order.",
                )

        if not isinstance(alpha, float) or not (0 <= alpha <= 1):
            raise CONFETTIConfigurationError(
                message=f"alpha must be a float in [0, 1], got {alpha!r}.",
                param="alpha",
            )
        if not isinstance(theta, float) or not (0 < theta < 1):
            raise CONFETTIConfigurationError(
                message=f"theta must be a float in (0, 1), got {theta!r}.",
                param="theta",
            )

        for name, value in [
            ("n_partitions", n_partitions),
            ("population_size", population_size),
            ("maximum_number_of_generations", maximum_number_of_generations),
        ]:
            if not isinstance(value, int) or value < 1:
                raise CONFETTIConfigurationError(
                    message=f"{name} must be a positive integer, got {value!r}.",
                    param=name,
                )

        for name, value in [
            ("crossover_probability", crossover_probability),
            ("mutation_probability", mutation_probability),
        ]:
            if not isinstance(value, float) or not (0 <= value <= 1):
                raise CONFETTIConfigurationError(
                    message=f"{name} must be a float in [0, 1], got {value!r}.",
                    param=name,
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

        if categorical_features is not None:
            if not isinstance(categorical_features, list) or not all(isinstance(i, int) for i in categorical_features):
                raise CONFETTIDataTypeError(
                    message="categorical_features must be a list of integers.",
                    param="categorical_features",
                    hint="Pass a list of column indices, e.g. [0, 3, 5].",
                )
            if len(categorical_features) != len(set(categorical_features)):
                raise CONFETTIConfigurationError(
                    message="categorical_features contains duplicate indices.",
                    param="categorical_features",
                )
            n_features = inst_shape[1]
            for idx in categorical_features:
                if idx < 0 or idx >= n_features:
                    raise CONFETTIConfigurationError(
                        message=f"categorical_features index {idx} is out of range for {n_features} features.",
                        param="categorical_features",
                        hint=f"Indices must be in [0, {n_features - 1}].",
                    )

        if relation_constraints is not None:
            if not isinstance(relation_constraints, list):
                raise CONFETTIDataTypeError(
                    message="relation_constraints must be a list of RelationConstraint instances.",
                    param="relation_constraints",
                )
            for i, c in enumerate(relation_constraints):
                if not isinstance(c, RelationConstraint):
                    raise CONFETTIDataTypeError(
                        message=f"relation_constraints[{i}] is {type(c).__name__}, expected RelationConstraint.",
                        param="relation_constraints",
                    )

        if processes is not None:
            if not isinstance(processes, int) or processes < 1:
                raise CONFETTIConfigurationError(
                    message=f"processes must be a positive integer or None, got {processes!r}.",
                    param="processes",
                )
