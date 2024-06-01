import numpy as np
from scipy.interpolate import RegularGridInterpolator

from CoolDwarf.utils.interp import linear_interpolate_ndarray
from CoolDwarf.utils.interp import find_closest_values

class OPACInterp:
    def __init__(self, opacDict, X, Z):
        self.X = X
        self.Z = Z
        self._opacDict = opacDict
        self._tempKeys = opacDict[0.0][0.0]['LogT']
        self._densKeys = opacDict[0.0][0.0]['LogR']
        self._interpolate_tables()
        print(self._tempKeys, self._densKeys)
        self._kappaLookup = RegularGridInterpolator((self._tempKeys, self._densKeys), self._opac, bounds_error=False)

    def _interpolate_tables(self):
        Xupper, Xlower = find_closest_values(list(self._opacDict.keys()), self.X)
        Zupper, Zlower = find_closest_values(list(self._opacDict[0.0].keys()), self.Z)
        if Xlower == None and Zlower == None:
            self._opac = self._opacDict[Xupper][Zupper]['Kappa']
        elif Xlower == None and Zlower != None:
            opacUpperZ = self._opacDict[Xupper][Zupper]['Kappa']
            opacLowerZ = self._opacDict[Xupper][Zlower]['Kappa']
            self._opac = linear_interpolate_ndarray((opacLowerZ, opacUpperZ), (Zlower, Zupper), self.Z)
        elif Xlower != None and Zlower == None:
            opacUpperX = self._opacDict[Xupper][Zupper]['Kappa']
            opacLowerX = self._opacDict[Xlower][Zupper]['Kappa']
            self._opac = linear_interpolate_ndarray((opacLowerX, opacUpperX), (Xlower, Xupper), self.X)
        else:
            opacUpperXUpperZ = self._opacDict[Xupper][Zupper]['Kappa']
            opacUpperXLowerZ = self._opacDict[Xupper][Zlower]['Kappa']
            opacLowerXUpperZ = self._opacDict[Xlower][Zupper]['Kappa']
            opacLowerXLowerZ = self._opacDict[Xlower][Zlower]['Kappa']
            opacUpperX = linear_interpolate_ndarray((opacUpperXLowerZ, opacUpperXUpperZ), (Zlower, Zupper), self.Z)
            opacLowerX = linear_interpolate_ndarray((opacLowerXLowerZ, opacLowerXUpperZ), (Zlower, Zupper), self.Z)
            self._opac = linear_interpolate_ndarray((opacLowerX, opacUpperX), (Xlower, Xupper), self.X)

    def kappa(self, temp, density):
        T6 = temp * 1e-6
        logR = np.log10(density) -  3*np.log10(T6)
        logT = np.log10(temp)
        return self._kappaLookup((logT, logR))
