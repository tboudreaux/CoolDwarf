import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from CoolDwarf.utils import linear_interpolate_dataframes


class EOS:
    def __init__(self, tablePath):
        self._tablePath = tablePath
        self.parse_table()

    def parse_table(self):
        tableExtract = re.compile(r"(#iT=\s*\d+\slog T=\s*(\d+\.\d+))\n(((\s+(?:-?)\d\.\d+E[+-]\d+){10}\n?)*)")
        with open(self._tablePath, 'r') as f:
            content = f.read()
        dataSection = '\n'.join(content.split('\n')[1:])
        columns = ["logT", "logP", "logRho", "logU", "logS", "dlrho/dlT_P", "dlrho/dlP_T", "dlS/dlT_P", "dlS/dlP_T", "grad_ad"]
        self._EOSTabs = dict()
        for match in re.finditer(tableExtract, dataSection):
            logT = float(match.groups()[1])
            table = match.groups()[2]
            self._EOSTabs[logT] = pd.read_fwf(StringIO(table), colspec='infer', names=columns)
        self._temps = np.array(list(self._EOSTabs.keys()))

    def __call__(self, logT, logRho, target="pressure"):
        if not self._temps.min() <= logT <= self._temps.max():
            raise ValueError(f"Temperature is not in bounds of EOS table -- {logT} ∉ ({self._temps.min():0.3f}, {self._temps.max():0.3f})")
        targetTempEOS = linear_interpolate_dataframes(self._EOSTabs, logT)
        lookup = {
            "pressure": targetTempEOS.logP.values,
            "U": targetTempEOS.logU.values
        }
        if target not in lookup:
            raise KeyError(f"{target} is not a valid interpolation target")
        F = interp1d(targetTempEOS.logRho.values, lookup[target])
        return 10**F(logRho)
