"""
Ferguson et al. (2005) Opacity Table module. This module takes a single file which is
the merging of an entire Ferguson opacity table composition set and parses it. At 
instantiation a table is interpolated to the correct X and Z from the provided
table set. The table is then used to calculate the opacity at a given
logR and logT. Currently this uses the GS98 composition file. This can be extended
if you are interested in extending it to other compositions.

..math::
    R  = \\rho \\left(\\frac{T}{10^6}\\right)^3

Functions
=========
linear_interpolator(x, x1, x2, Q1, Q2)
    Provides a linear interpolation between two points

bilinear_interpolator(x, z, x1, x2, z1, z2, Q00, Q10, Q01, Q11)
    Provides a bilinear interpolation between four points

find_closest_indices(x, z, keys)
    Finds the closest indices in a dictionary to a given x and z

Classes
=======
Ferg05Opacity
    Class for the Ferguson et al. (2005) opacity table
    It is used to calculate the opacity at a given logR and log
    and provides the standard CoolDwarf opacity interface (the
    kappa function)

Example Usage
=============
>>> from CoolDwarf.opac.ferg05.opac import Ferg05Opacity
>>> X, Z = 0.7, 0.02
>>> opac = Ferg05Opacity(X, Z)
>>> logKappa = opac.kappa(4e3, 1e-1)
>>> print(f"Kappa_rmo: {10**logKappa} cm^2/g")

"""
import re
import importlib.resources as pkg
import pandas as pd

import logging
import numpy as np

from CoolDwarf.opac.ferg05 import include
from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.utils.interp.interpolate import bilinear_interpolator, find_closest_indices

xp, CUPY = get_array_module()
if CUPY:
    from cupyx.scipy.interpolate import RegularGridInterpolator
else:
    from scipy.interpolate import RegularGridInterpolator

PARSE = re.compile(r".+\s+\d{4}\swith\sX=\s(\d\.\d+)\sand\sZ=\s(\d\.\d+)\n\s+log\sR\n\s\nlog\sT\s+((-{0,1}\d+\.\d+\s{1,2}\n?){19})(((-{0,1}\d+\.\d+\s*\n?){20}){84})")

class Ferg05Opacity():
    def __init__(self, X, Z):
        self.X = X
        self.Z = Z
        self._logger = logging.getLogger(__name__)

        opacTables = self.parse()
        self.opacTable, self.interpF = self.interpolate(opacTables)
        self._tempBounds = (self.temps.min(), self.temps.max())
        self._logRBounds = (self.logRs.min(), self.logRs.max())

    def parse(self):
        files = pkg.files(include)
        path = files / "GS98.F05"
        with open(path, "r") as f:
            contents = f.read()

        tables = PARSE.finditer(contents)
        opacTables = dict()
        self.temps = list()
        for tableID, table in enumerate(tables):
            X = float(table.group(1))
            Z = float(table.group(2))
            data = table.group(5)
            data = data.replace('-', ' -')
            dataLines = data.split('\n')
            dataLinesNumeric = list()
            for line in dataLines:
                lineList = list()
                for num in line.split():
                    lineList.append(float(num))
                if len(lineList) != 0:
                    dataLinesNumeric.append(lineList[1:])
                    if tableID == 0:
                        self.temps.append(float(lineList[0]))
            if tableID == 0:
                self.logRs = xp.array([float(x) for x in table.group(3).split()])
                self.temps = xp.array(self.temps)


            opacTables[(X, Z)] = pd.DataFrame(dataLinesNumeric, columns=None)

        return opacTables

    def kappa(self, temp, density, log=False):
        temp, density = xp.array(temp), xp.array(density)
        targetR = density * (temp/1e6)**3
        targetLogR = np.log10(targetR)
        targetLogT = np.log10(temp)
        if xp.any(targetLogT < self._tempBounds[0]):
            self._logger.warning(f"Temperature out of bounds (below) (LTO): Bounds = {self._tempBounds[0]} K - {self._tempBounds[1]} K")
            self.interpF.bounds_error = False
            self.interpF.fill_value = -1
        if xp.any(targetLogT > self._tempBounds[1]):
            self._logger.warning(f"Temperature out of bounds (above) (LTO): Bounds = {self._tempBounds[0]} K - {self._tempBounds[1]} K")
            self.interpF.bounds_error = False
            self.interpF.fill_value = None
        if xp.any(targetLogR < self._logRBounds[0]):
            self._logger.warning(f"logR out of bounds (below) (LTO): log R Bounds = {self._logRBounds[0]} - {self._logRBounds[1]}")
            self.interpF.bounds_error = False
            self.interpF.fill_value = -1
        if xp.any(targetLogR > self._logRBounds[1]):
            self._logger.warning(f"logR out of bounds (above) (LTO): log R Bounds = {self._logRBounds[0]} - {self._logRBounds[1]}")
            self.interpF.bounds_error = False
            self.interpF.fill_value = None
        logKappa = self.kappaLogRT(xp.array([targetLogT]), xp.array([targetLogR]))
        self.interpF.bounds_error = True
        self.interpF.fill_value = xp.nan
        if not log:
            return xp.reshape(10**logKappa, temp.shape)
        return xp.reshape(logKappa, temp.shape)

    def check_bounds(self, temp, density, log=False):
        ...

    def kappaLogRT(self, logT, logR):
        return self.interpF((logR, logT))


    def interpolate(self, dataframes):
        x1, x2, z1, z2 = find_closest_indices(self.X, self.Z, dataframes.keys())
        if x1 == x2 == self.X and z1 == z2 == self.Z:
            self._logger.info("X and Z values found in Ferg05 table")
            table = xp.array(dataframes[(self.X, self.Z)].values)
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
        interpF = RegularGridInterpolator((self.logRs, self.temps), table.T, method='linear', bounds_error=False, fill_value=None)
        return table, interpF

if __name__ == "__main__":
    opac = Ferg05Opacity(0.0, 0.0)
    print(opac.kappa(10000, 1e4))
    print(opac.kappaLogRT(xp.array([4.5]), xp.array([-8.0])))
