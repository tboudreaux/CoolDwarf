import re
from io import StringIO
import numpy as np

def load_lowtempopac(path : str) -> dict:
    tablePattern = re.compile(r".+X=\s(\d\.\d+)\sand\sZ=\s(\d\.\d+)\s+log R\s+\slog T((?:\s+-?\d\.\d+){19})\s+((?:-?\d+\.\d+\s*){1700})")
    with open(path, 'r') as f:
        content = f.read()

    tables = dict()
    for table in re.finditer(tablePattern, content):
        X = float(table.groups()[0])
        Z = float(table.groups()[1])
        logRIO = StringIO(table.groups()[2])
        tableIO = StringIO(table.groups()[3])

        tableNP = np.genfromtxt(tableIO, delimiter=[5, 8, *[7]*18])
        logT = tableNP[:, 0]
        logR = np.genfromtxt(logRIO)
        kappa = tableNP[:, 1:]

        tables[Z] = {
                'X' : X,
                'Z' : Z,
                'LogT' : logT,
                'LogR' : logR,
                'Kappa': kappa
                }
    return tables
