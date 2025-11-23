from typing import Any, Mapping, Optional


class CONFETTIError(Exception):
    """
    Base class for all CONFETTI errors.

    Parameters
    ----------
    message
        Description of the configuration problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """

    message: str
    config: Optional[Mapping[str, Any]]
    param: Optional[str]
    hint: Optional[str]
    source: Optional[str]

    def __init__(
        self,
        message: str,
        config: Optional[Mapping[str, Any]] = None,
        param: Optional[str] = None,
        hint: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.config = config
        self.param = param
        self.hint = hint
        self.source = source


class CONFETTIConfigurationError(CONFETTIError):
    """
    Error to raise when users input a configuration that is not compatible with CONFETTI's design.

    Parameters
    ----------
    message
        Description of the configuration problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """

    def __init__(
        self,
        message: str,
        config: Optional[Mapping[str, Any]] = None,
        param: Optional[str] = None,
        hint: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        super().__init__(message, config, param, hint, source)


class CONFETTIDataTypeError(CONFETTIError):
    """
    Error to raise when there is a data type mismatch in CONFETTI's operations.

    Parameters
    ----------
    message
        Description of the configuration problem.
    config
        The configuration mapping that triggered the error, or a relevant subset.
    param
        The specific configuration key that is invalid, if known.
    hint
        Optional remediation advice that helps the user fix the issue, if known.
    source
        Where the configuration came from.
    """

    def __init__(
        self,
        message: str,
        config: Optional[Mapping[str, Any]] = None,
        param: Optional[str] = None,
        hint: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        super().__init__(message, config, param, hint, source)
