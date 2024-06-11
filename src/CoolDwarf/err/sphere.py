class VolumeError(Exception):
    """
    An exception class for volume error. This exception is raised when the volume is not valid.
    """
    def __init__(self, msg):
        self.message = msg

class ResolutionError(Exception):
    """
    An exception class for volume error. This exception is raised when the volume is not valid.
    """
    def __init__(self, msg):
        self.message = msg

