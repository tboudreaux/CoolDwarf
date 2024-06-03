class EOSFormatError(Exception):
    def __init__(self, asked, keys):
        msg = f"{asked} is not a EOS format key. Valid keys are {', '.join(list(keys))}"
        self.message = msg

