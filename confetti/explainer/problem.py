from pymoo.core.problem import ElementwiseProblem, Problem
import numpy as np


class CounterfactualProblem(Problem):
    def __init__(self, original_instance: np.array, nun_instance: np.array, nun_index: int, start_timestep: int,
                 subsequence_length: int, classifier, y_pred_train, alpha=0.5, theta = 0.51):
        self.original_instance = original_instance
        self.nun_instance = nun_instance
        self.nun_index = nun_index
        self.start_timestep = start_timestep
        self.end_timestep = start_timestep + subsequence_length
        self.classifier = classifier
        self.y_pred_train = y_pred_train
        self.subsequence_length = subsequence_length
        self.alpha = alpha #Parameter to control the trade-off between precision and sparsity
        self.theta = theta #Threshold for confidence
        n_var = original_instance.shape[1] * (self.subsequence_length)  # Here, shape[1] are the no. of channels

        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Here, x represents a single possible counterfactual explanation
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

        #x_reshaped = x.reshape(self.subsequence_length, self.original_instance.shape[1])
        #counterfactual = np.copy(self.original_instance)
        #counterfactual[self.start_timestep:self.end_timestep][x_reshaped] = \
        #self.nun_instance[self.start_timestep:self.end_timestep][x_reshaped]

        # Reshape counterfactual correctly as expected by classifier
        #counterfactual = counterfactual.reshape(1, counterfactual.shape[0], counterfactual.shape[1])

        # Confidence Formula
        f1 = self.classifier.predict(counterfactuals)[:, self.y_pred_train[self.nun_index]]
        # Sparsity Formula
        f2 = np.sum(x_reshaped, axis=(1, 2))


        # Objective values. Maximize f1 and minimize f2
        out["F"] = [-self.alpha * f1, (1-self.alpha) * f2]

        # Confidence Constraint. Confidence needs to be at least 51%
        out["G"] = [self.theta - f1]