import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from io import StringIO

from CoolDwarf.utils.interp import linear_interpolate_dataframes


class CH21EOS:
    def __init__(self, tablePath):
        self._tablePath = tablePath
        self.parse_table()

    def parse_table(self):
        tableExtract = re.compile(r"(#iT=\s*\d+\slog T=\s*(\d+\.\d+))\n(((\s+(?:-?)\d\.\d+E[+-]\d+){10}\n?)*)")
        with open(self._tablePath, 'r') as f:
            content = f.read()
        dataSection = '\n'.join(content.split('\n')[1:])
        columns = ["logT", "logP", "logRho", "logU", "logS", "dlrho/dlT_P", "dlrho/dlP_T", "dlS/dlT_P", "dlS/dlP_T", "grad_ad"]
        self._EOSTabs = list()
        self._temps = list()
        for match in re.finditer(tableExtract, dataSection):
            logT = float(match.groups()[1])
            self._temps.append(logT)
            table = match.groups()[2]
            df = pd.read_fwf(StringIO(table), colspec='infer', names=columns)
            self._EOSTabs.append(df.values)
        self._temps = np.array(self._temps)
        self._EOSTabs = np.array(self._EOSTabs)
        self._rhos = self._EOSTabs[0, :, 2]

        self._forward_pressure = RegularGridInterpolator((self._temps, self._rhos), self._EOSTabs[:, :, 1])
        self._forward_energy = RegularGridInterpolator((self._temps, self._rhos), self._EOSTabs[:, :, 3])

    def check_forward_params(self, logT, logRho):
        mask = (self._temps.min() <= logT) & (logT <= self._temps.max())
        if not mask.all():
            raise ValueError(f"Temperature (log10T) is not in bounds of EOS table -- {logT} ∉ ({self._temps.min():0.3f}, {self._temps.max():0.3f})")
        mask = (self._rhos.min() <= logRho) & (logRho <= self._rhos.max())
        if not mask.all():
            raise ValueError(f"Density (log10Rho) is not in bounds of EOS table -- {logRho} ∉ ({self._rhos.min():0.3f}, {self._rhos.max():0.3f})")

    def pressure(self, logT, logRho):
        self.check_forward_params(logT, logRho)
        return 10**self._forward_pressure((logT, logRho))

    def energy(self, logT, logRho):
        self.check_forward_params(logT, logRho)
        return 10**self._forward_energy((logT, logRho))

