from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

class OpacitySet:
    def __init__(self, low, high):
        self.low = low
        self.high = high

        self.boundary = (3.75, 4.5)

    def kappa(self, temp, density, log=False):
        if not log:
            checkTemp = xp.log10(temp)
        else:
            checkTemp = temp
        if checkTemp < self.boundary[0]:
            return self.low.kappa(temp, density, log=log)
        elif checkTemp > self.boundary[1]:
            return self.high.kappa(temp, density, log=log)
        else:
            fh = (4/3) * (temp - self.boundary[0])
            fl = 1 - fh
            assert fh + fl == 1, "Ramp Error! fh + fl != 1. This probably means the temp being passed into the opacity set is log logged right"

            return fl * self.low.kappa(temp, density) + fh * self.high.kappa(temp, density)

