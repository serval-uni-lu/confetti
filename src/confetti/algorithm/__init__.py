"""
CONFETTI's own NSGA-III genetic algorithm.

This package replaces pymoo with hand-rolled implementations, paving
the way for an eventual Rust / PyO3 backend.
"""

from confetti.algorithm._nsga3 import NSGA3, Result, minimize
from confetti.algorithm._operators import Crossover, Mutation, Sampling
from confetti.algorithm._problem import Problem
from confetti.algorithm.crossover import TwoPointCrossover
from confetti.algorithm.mutation import BitflipMutation
from confetti.algorithm.reference_directions import das_dennis
from confetti.algorithm.sampling import BinaryRandomSampling

__all__ = [
    "BinaryRandomSampling",
    "BitflipMutation",
    "Crossover",
    "Mutation",
    "NSGA3",
    "Problem",
    "Result",
    "Sampling",
    "TwoPointCrossover",
    "das_dennis",
    "minimize",
]
