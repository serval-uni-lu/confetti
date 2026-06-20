from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("confetti-ts")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .explainer.explainer import CONFETTI
from .errors import CONFETTIConfigurationError, CONFETTIError, CONFETTIDataTypeError
from confetti.structs.counterfactual_structs import Counterfactual, CounterfactualSet, CounterfactualResults


__all__ = [
    "CONFETTI",
    "CONFETTIError",
    "CONFETTIConfigurationError",
    "CONFETTIDataTypeError",
    "Counterfactual",
    "CounterfactualSet",
    "CounterfactualResults",
]
