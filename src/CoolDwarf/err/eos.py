class EOSFormatError(Exception):
    """
    An exception class for EOS format error. This exception is raised when the EOS format is not recognized.
    """
    def __init__(self, msg):
        self.message = msg

class EOSInverterError(Exception):
    """
    An exception class for EOS inverter error. This exception is raised when an error occurs during the inversion of the EOS.
    """
    def __init__(self, msg):
        self.message = msg

class EOSBoundsError(Exception):
    """
    An exception class for EOS bounds error. This exception is raised when the bounds for the inversion are not valid.
    """
    def __init__(self, msg):
        self.message = msg

