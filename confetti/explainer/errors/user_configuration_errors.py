class CONFETTIConfigurationError(Exception):
    """
    Error to raise when users input a configuration that is not compatible with CONFETTI's design.

    Args:
        Exception (_type_): _description_
    """

    def __init__(self, message: str = "Invalid Configuration"):
        super().__init__(message)
        self.message = message