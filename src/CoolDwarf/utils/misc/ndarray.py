import numpy as np

class CallbackNDArray(np.ndarray):
    def __new__(cls, input_array, callback=None, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        obj.callback = callback
        return obj

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if hasattr(self, 'callback') and self.callback:
            self.callback()
