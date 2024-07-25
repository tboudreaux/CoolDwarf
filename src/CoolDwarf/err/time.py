class TimestepError(Exception):
    """
    An exception class for volume error. This exception is raised when the volume is not valid.
    """
    def __init__(self, msg):
        self.message = msg


