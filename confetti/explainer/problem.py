from pymoo.core.problem import ElementwiseProblem
import numpy as np


class CounterfactualProblem(ElementwiseProblem):
    def __init__(self, original_instance: np.array, nun_instance: np.array, nun_index: int, start_timestep: int,
                 subsequence_length: int, classifier, y_pred_train):
        self.original_instance = original_instance
        self.nun_instance = nun_instance
        self.nun_index = nun_index
        self.start_timestep = start_timestep
        self.end_timestep = start_timestep + subsequence_length
        self.classifier = classifier
        self.y_pred_train = y_pred_train
        self.subsequence_length = subsequence_length
        n_var = original_instance.shape[1] * (self.subsequence_length)  # Here, shape[1] are the no. of channels

        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Here, x represents a single possible counterfactual explanation
        """
        x_reshaped = x.reshape(self.subsequence_length, self.original_instance.shape[1])
        counterfactual = np.copy(self.original_instance)
        counterfactual[self.start_timestep:self.end_timestep][x_reshaped] = \
        self.nun_instance[self.start_timestep:self.end_timestep][x_reshaped]

        # Reshape counterfactual correctly as expected by classifier
        counterfactual = counterfactual.reshape(1, counterfactual.shape[0], counterfactual.shape[1])

        # Precision Formula
        f1 = self.classifier.predict(counterfactual)[0][self.y_pred_train[self.nun_index]]
        # Sparsity Formula
        f2 = np.mean(self.original_instance.flatten() != counterfactual.flatten())

        # Objective values. Maximize f1 and minimize f2
        out["F"] = [-f1, f2]

        # Precision Constraint. Precision needs to be at least 51%
        out["G"] = [0.51 - f1]