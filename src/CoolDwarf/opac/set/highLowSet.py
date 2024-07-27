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
        checkTemp = checkTemp.flatten()
        # linear ramp to scale between low and high opacity tables
        # the ramp is such that at logT = 3.75 the full weight is
        # given to the low temp opac tables and then it scales linearly
        # from logT = 3.75 to logT = 4.5 such that by log T = 4.5
        # the full weight is given to the high temp opac tables.
        fh = (4/3) * (checkTemp - self.boundary[0])
        fl = 1 - fh

        if isinstance(temp, float) and isinstance(density, float):
            if checkTemp < self.boundary[0]:
                return self.low.kappa(temp, density, log=log)
            elif checkTemp > self.boundary[1]:
                return self.high.kappa(temp, density, log=log)
            else:
                return fl * self.low.kappa(temp, density) + fh * self.high.kappa(temp, density)
        elif isinstance(temp, xp.ndarray) and isinstance(density, xp.ndarray):
            lowTInRange = temp[checkTemp < self.boundary[0]]
            lowDInRange = density[checkTemp < self.boundary[0]]
            highTInRange = temp[checkTemp > self.boundary[1]]
            highDInRange = density[checkTemp > self.boundary[1]]
            midTRange = temp[(checkTemp >= self.boundary[0]) & (checkTemp <= self.boundary[1])]
            midDRange = density[(checkTemp >= self.boundary[0]) & (checkTemp <= self.boundary[1])]
            # linear ramp to scale between low and high opacity tables
            # the ramp is such that at logT = 3.75 the full weight is
            # given to the low temp opac tables and then it scales linearly
            # from logT = 3.75 to logT = 4.5 such that by log T = 4.5
            # the full weight is given to the high temp opac tables.
            fh = (4/3) * (checkTemp - self.boundary[0])
            fl = 1 - fh
            fh = fh[(checkTemp >= self.boundary[0]) & (checkTemp <= self.boundary[1])]
            fl = fl[(checkTemp >= self.boundary[0]) & (checkTemp <= self.boundary[1])]

            fh = xp.reshape(fh, midTRange.shape)
            fl = xp.reshape(fl, midTRange.shape)

            lowTempOpac = self.low.kappa(lowTInRange, lowDInRange)
            highTempOpac = self.high.kappa(highTInRange, highDInRange)
            midTempOpac = fl * self.low.kappa(midTRange, midDRange) + fh * self.high.kappa(midTRange, midDRange)
            outOpac = xp.zeros_like(temp)
            outOpac[checkTemp < self.boundary[0]] = lowTempOpac
            outOpac[checkTemp > self.boundary[1]] = highTempOpac
            outOpac[(checkTemp >= self.boundary[0]) & (checkTemp <= self.boundary[1])] = midTempOpac
            
            return outOpac
        else:
            raise TypeError("Input density and/or temperature type is not supported (both float or both ndarrays)")

