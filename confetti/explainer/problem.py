from pymoo.core.problem import ElementwiseProblem, Problem
import numpy as np


class CounterfactualProblem(Problem):
    def __init__(self,
                 original_instance: np.array,
                 nun_instance: np.array,
                 nun_index: int,
                 start_timestep: int,
                 subsequence_length: int,
                 classifier,
                 reference_labels: np.array,
                 alpha: float = 0.5,
                 theta: float = 0.51):
        """
        Define a multi-objective optimization problem for generating counterfactuals
        using a perturbed subsequence of the original time series instance.

        This class is designed to be used with evolutionary algorithms from pymoo. It encodes
        the counterfactual generation process as a two-objective optimization problem, where
        the goals are to maximize confidence in the target class and minimize the number of changes (sparsity).
        A third constraint ensures that the predicted probability exceeds a given threshold (`theta`).

        Parameters:
        ----------
        original_instance : np.array
            The original multivariate time series instance to be explained.
            Shape: (timesteps, channels)
        nun_instance : np.array
            The Nearest Unlike Neighbour (NUN) used to guide counterfactual generation.
        nun_index : int
            Index of the NUN in the reference dataset.
        start_timestep : int
            Starting index of the subsequence to be perturbed.
        subsequence_length : int
            Length of the subsequence within the time series to be modified.
        classifier : object
            Trained classifier with a `predict_proba` method.
        reference_labels : np.array
            Labels for the reference dataset, used to evaluate class change.
        alpha : float, default=0.5
            Trade-off parameter between sparsity and confidence.
            A higher value puts more weight on achieving confident predictions.
        theta : float, default=0.51
            Confidence threshold for the target class (i.e., predicted probability must be ≥ theta).

        Attributes:
        ----------
        n_var : int
            Number of decision variables (equal to channels × subsequence_length).
        n_obj : int
            Number of objectives (confidence and sparsity).
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
        self.alpha = alpha
        self.theta = theta
        n_var = original_instance.shape[1] * (self.subsequence_length)  # Here, shape[1] are the no. of channels

        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Here, x represents a batch of possible counterfactual explanations
        """
        n_samples = x.shape[0]
        n_features = self.original_instance.shape[1]
        # Reshape the mask to (n_samples, subsequence_length, n_features)
        x_reshaped = x.reshape(n_samples, self.subsequence_length, n_features)

        # Repeat original instance across all samples
        counterfactuals = np.tile(self.original_instance, (n_samples, 1, 1))

        # Extract the patch to replace from nun_instance (shared for all samples)
        replacement_patch = self.nun_instance[
                            self.start_timestep:self.end_timestep]  # shape: (subsequence_length, n_features)

        # Broadcast replacement_patch to match number of samples
        replacement_patch_broadcasted = np.broadcast_to(replacement_patch,
                                                        (n_samples, self.subsequence_length, n_features))

        # Apply mask-based replacement
        counterfactuals[:, self.start_timestep:self.end_timestep, :] = (
            np.where(x_reshaped, replacement_patch_broadcasted,
                     counterfactuals[:, self.start_timestep:self.end_timestep, :])
        )


        # Confidence Formula
        f1 = self.classifier.predict(counterfactuals)[:, self.reference_labels[self.nun_index]]
        # Sparsity Formula
        f2 = np.sum(x_reshaped, axis=(1, 2))


        # Objective values. Maximize f1 and minimize f2
        out["F"] = [-self.alpha * f1, (1-self.alpha) * f2]

        # Confidence Constraint. Confidence needs to be at least theta
        out["G"] = [self.theta - f1]