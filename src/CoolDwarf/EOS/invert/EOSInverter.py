from scipy.optimize import minimize
import numpy as np

from CoolDwarf.err import EOSInverterError

class Inverter:
    def __init__(self, EOS, TRange, RhoRange):
        self.EOS = EOS
        self.TRange = TRange
        self.RhoRange = RhoRange
        self._bounds = (self.TRange, self.RhoRange)

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
