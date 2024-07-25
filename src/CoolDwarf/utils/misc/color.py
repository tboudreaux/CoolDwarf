from dataclasses import dataclass

@dataclass
class ANSIIColor:
    BLACK: str = "\033[30m"
    RED: str = "\033[31m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    BLUE: str = "\033[34m"
    MAGENTA: str = "\033[35m"
    CYAN: str = "\033[36m"
    WHITE: str = "\033[37m"
    RESET: str = "\033[0m"
    def __getitem__(self, key):
        return getattr(self, key)
    def __iter__(self):
        return iter([self.BLACK, self.RED, self.GREEN, self.YELLOW, self.BLUE, self.MAGENTA, self.CYAN, self.WHITE, self.RESET])

    def get_color(self, key):
        return getattr(self, key)

def color_string(string: str, color: str):
    colors = ANSIIColor()
    return f"{colors.get_color(color.upper())}{string}{ANSIIColor.RESET}"
