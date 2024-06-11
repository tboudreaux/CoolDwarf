class EnergyConservationError(Exception):
    def __init__(self, msg):
        self.message = msg

class NonConvergenceError(Exception):
    def __init__(self, msg):
        self.message = msg
