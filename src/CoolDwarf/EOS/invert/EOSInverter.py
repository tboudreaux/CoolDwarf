from scipy.optimize import minimize
import numpy as np

from CoolDwarf.err import EOSInverterError, EOSBoundsError

class Inverter:
    def __init__(self, EOS, TRange, RhoRange):
        self.EOS = EOS
        self._validate_bounds((TRange, RhoRange))
        self._TRange = TRange
        self._RhoRange = RhoRange
        self._bounds = (self._TRange, self._RhoRange)

    def temperature_density(self, energy, logTInit, logRhoInit):
        x0 = [logTInit, logRhoInit]
        r = minimize(self._loss, x0, energy, bounds=self._bounds, method="Nelder-Mead")
        if not r.success:
            raise EOSInverterError(f"No Inversion found for U={energy:0.3f} within (logT, logRho) = {self._bounds}")
        logT = r.x[0]
        logRho = r.x[1]
        return np.array([logT, logRho])

    def _loss(self, x, target):
        logT = x[0]
        logRho = x[1]
        l = np.abs(self.EOS.energy(logT, logRho) - target)
        return l

    def set_bounds(self, newBounds):
        self._validate_bounds(newBounds)
        self._TRange = newBounds[0]
        self._RhoRange = newBounds[1]
        self._bounds = (self._TRange, self._RhoRange)
        

    def _validate_bounds(self, bounds):
        Tl, Th = bounds[0][0], bounds[0][1]
        Rl, Rh = bounds[1][0], bounds[1][1]

        try:
            assert self.EOS._temps.min() <= Tl <= self.EOS._temps.max()
            assert self.EOS._temps.min() <= Th <= self.EOS._temps.max()
            assert Tl < Th
            assert self.EOS._rhos.min() <= Rl <= self.EOS._rhos.max()
            assert self.EOS._rhos.min() <= Rh <= self.EOS._rhos.max()
            assert Rl < Rh
        except AssertionError as e:
            raise EOSBoundsError(f"Invalid Bounds provided to EOS inverter")
