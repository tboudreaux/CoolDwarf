class EnergyConservationError(Exception):
    """
    An exception class for energy conservation error. This exception is raised when the energy conservation is not
    satisfied during integration of the model.
    """
    def __init__(self, msg):
        self.message = msg

class NonConvergenceError(Exception):
    """
    An exception class for non-convergence error. This exception is raised when the solver does not converge.
    """
    def __init__(self, msg):
        self.message = msg
