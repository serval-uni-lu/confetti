"""Abstract base class for optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Problem(ABC):
    """
    Base class for multi-objective optimization problems.

    Subclasses must implement ``_evaluate`` which populates an output
    dictionary with objective values (``"F"``) and, optionally,
    inequality-constraint values (``"G"``).

    Parameters
    ----------
    ``n_var`` : int
        Number of decision variables.
    ``n_obj`` : int
        Number of objectives.
    ``n_ieq_constr`` : int
        Number of inequality constraints (``G ≤ 0`` convention).
    """

    def __init__(self, n_var: int, n_obj: int, n_ieq_constr: int = 0) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of solutions.

        Calls the user-defined ``_evaluate`` and normalises the output
        shapes so downstream code can rely on consistent array layouts.

        Parameters
        ----------
        ``X`` : np.ndarray
            Decision-variable matrix of shape ``(n, n_var)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(F, G)`` where ``F`` has shape ``(n, n_obj)`` and ``G``
            has shape ``(n, n_ieq_constr)``.
        """
        out: dict[str, Any] = {}
        self._evaluate(X, out)

        F = np.asarray(out["F"], dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(-1, 1)

        G_raw = out.get("G")
        if G_raw is not None:
            G = np.asarray(G_raw, dtype=np.float64)
            # CounterfactualProblem produces out["G"] = [theta - f1] which
            # yields shape (1, n).  Normalise to (n, n_ieq_constr).
            if G.ndim == 1:
                G = G.reshape(-1, 1)
            if G.shape[0] == 1 and G.shape[1] == X.shape[0]:
                G = G.T
        else:
            G = np.zeros((X.shape[0], max(self.n_ieq_constr, 1)))

        return F, G

    @abstractmethod
    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """
        Populate *out* with objective and constraint values.

        Parameters
        ----------
        ``x`` : np.ndarray
            Decision-variable matrix of shape ``(n, n_var)``.
        ``out`` : dict[str, Any]
            Output dictionary.  Must set ``out["F"]`` (shape
            ``(n, n_obj)``) and optionally ``out["G"]``.
        """
