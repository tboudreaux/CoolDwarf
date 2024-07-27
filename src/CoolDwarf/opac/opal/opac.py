"""
This module provides helper utilities for reading and parsing opal formated
opacity files.
"""

import re
import numpy as np
import importlib.resources as pkg

import logging

from CoolDwarf.opac.opal import include
from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.utils.interp.interpolate import bilinear_interpolator, find_closest_indices

xp, CUPY = get_array_module()
if CUPY:
    from cupyx.scipy.interpolate import RegularGridInterpolator
else:
    from scipy.interpolate import RegularGridInterpolator


class OPALHighTempOpacity:
    def __init__(self, X, Z):
        self.X = X
        self.Z = Z
        self._logger = logging.getLogger(__name__)
        self.temps = [
            3.75, 3.80, 3.85, 3.90, 3.95, 4.00, 4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35,
            4.40, 4.45, 4.50, 4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
            5.05, 5.10, 5.15, 5.20, 5.25, 5.30, 5.35, 5.40, 5.45, 5.50, 5.55, 5.60, 5.65,
            5.70, 5.75, 5.80, 5.85, 5.90, 5.95, 6.00, 6.10, 6.20, 6.30, 6.40, 6.50, 6.60,
            6.70, 6.80, 6.90, 7.00, 7.10, 7.20, 7.30, 7.40, 7.50, 7.60, 7.70, 7.80, 7.90,
            8.00, 8.10, 8.30, 8.50, 8.70
        ]
        self.logRs = [
                -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5,
                -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0
                ]

        self.temps = np.array(self.temps)
        self.logRs = np.array(self.logRs)
        self._tempBounds = (self.temps.min(), self.temps.max())
        self._logRBounds = (self.logRs.min(), self.logRs.max())

        opacTables = self.parse()
        self.opacTable, self.interpF = self.interpolate(opacTables)


    def parse(self):
        """
        Parse the 126 tables out of a properly formated opal opacity table.
        This idetifies all lines starting with Table # after the
        summary section and uses those to index where the tables begin. Given that
        opal opacity tables are not square and that numpy can only handel
        rectangular data all rows are padded to the length of the longest row with
        np.nan. Therefore, nan(s) should be interprited as locations where the
        opacity table was undefined.
        """
        COMP = re.compile(r"X=(\d\.\d+).*Z=(\d\.\d+)")
        files = pkg.files(include)
        path = files / "GS98.opal"
        with open(path) as f:
            contents = f.read().split('\n')
        sIndex = contents.index('************************************ Tables ************************************')
        ident = re.compile(r"TABLE\s+#(?:\s+)?\d+\s+\d+\s+X=\d\.\d+\s+Y=\d\.\d+\s+Z=\d\.\d+(?:\s+)?dX1=\d\.\d+\s+dX2=\d\.\d+")
        I = filter(lambda x: bool(re.match(ident, x[1])) and x[0] > sIndex+1, enumerate(contents))
        I = list(I)
        parsedTables = list(map(lambda x: [[float(z) for z in y.split()[1:]] for y in x], map(lambda x: contents[x[0]+6:x[0]+76], I)))

        paddedParsed = [list(map(lambda x: np.pad(x, (0, 19-len(x)), mode='constant', constant_values=(1,np.nan)), j)) for j in parsedTables]
        p = np.array(paddedParsed)

        tables = dict()
        for table, ident in zip(p, I):
            comp = re.findall(COMP, ident[1])
            X = float(comp[0][0])
            Z = float(comp[0][1])
            tables[(X, Z)] = table

        return tables

    def kappa(self, temp, density, log=False):
        temp, density = xp.array(temp), xp.array(density)
        targetR = density * (temp/1e6)**3
        targetLogR = np.log10(targetR)
        targetLogT = np.log10(temp)
        if xp.any(targetLogT < self._tempBounds[0]):
            self._logger.warning(f"Temperature out of bounds (below) (HTO): Bounds = {self._tempBounds[0]} K - {self._tempBounds[1]} K")
            self.interpF.bounds_error = False
            self.interpF.fill_value = -1
        if xp.any(targetLogT > self._tempBounds[1]):
            self._logger.warning(f"Temperature out of bounds (above) (HTO): Bounds = {self._tempBounds[0]} K - {self._tempBounds[1]} K")
            self.interpF.bounds_error = False
            self.interpF.fill_value = None
        if xp.any(targetLogR < self._logRBounds[0]):
            self._logger.warning(f"logR out of bounds (below) (HTO): log R Bounds = {self._logRBounds[0]} - {self._logRBounds[1]}")
            self.interpF.bounds_error = False
            self.interpF.fill_value = -1
        if xp.any(targetLogR > self._logRBounds[1]):
            self._logger.warning(f"logR out of bounds (above) (HTO): log R Bounds = {self._logRBounds[0]} - {self._logRBounds[1]}")
            self.interpF.bounds_error = False
            self.interpF.fill_value = None
        logKappa = self.kappaLogRT(xp.array([targetLogT]), xp.array([targetLogR]))
        self.interpF.bounds_error = True
        self.interpF.fill_value = xp.nan
        if not log:
            return xp.reshape(10**logKappa, temp.shape)
        return xp.reshape(logKappa, temp.shape)

    def kappaLogRT(self, logT, logR):
        return self.interpF((logR, logT))


    def interpolate(self, dataframes):
        x1, x2, z1, z2 = find_closest_indices(self.X, self.Z, dataframes.keys())
        if x1 == x2 == self.X and z1 == z2 == self.Z:
            self._logger.info("X and Z values found in Ferg05 table")
            table = xp.array(dataframes[(self.X, self.Z)])
            interpF = RegularGridInterpolator((self.logRs, self.temps), table.T, method='linear')
            return table, interpF
        elif x1 == x2 == self.X:
            self._logger.info("X value found in Ferg05 table")
        elif z1 == z2 == self.Z:
            self._logger.info("Z value found in Ferg05 table")

        table = bilinear_interpolator(
                self.X,
                self.Z,
                x1,
                x2,
                z1,
                z2,
                dataframes[(x1, z1)],
                dataframes[(x2, z1)],
                dataframes[(x1, z2)],
                dataframes[(x2, z2)]
                )
        interpF = RegularGridInterpolator((self.logRs, self.temps), table.T, method='linear')
        return table, interpF

if __name__ == "__main__":
    opac = OPALHighTempOpacity(0.7, 0.02)


