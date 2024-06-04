class EOSFormatError(Exception):
    def __init__(self, msg):
        self.message = msg

class EOSInverterError(Exception):
    def __init__(self, msg):
        self.message = msg

class EOSBoundsError(Exception):
    def __init__(self, msg):
        self.message = msg

