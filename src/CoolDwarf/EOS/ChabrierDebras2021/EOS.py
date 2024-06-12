"""
EOS.py -- EOS class for Chabrier Debras 2021 EOS tables

This module contains the EOS class for the Chabrier Debras 2021 EOS tables. The class is designed to be used with the
CoolDwarf Stellar Structure code, and provides the necessary functions to interpolate the EOS tables.

As with all EOS classes in CoolDwarf, the CH21EOS class is designed to accept linear values in cgs units, and return
linear values in cgs units.

Dependencies
------------
- pandas
- scipy
- cupy
- torch

Example usage
-------------
>>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
>>> eos = CH21EOS("path/to/eos/table")
>>> pressure = eos.pressure(7.0, -2.0)
>>> energy = eos.energy(7.0, -2.0)
"""
import re
import pandas as pd
from io import StringIO
import logging

import torch
import numpy as np

from CoolDwarf.utils.misc.backend import get_interpolator, get_array_module

xp, CUPY = get_array_module()
RegularGridInterpolator = get_interpolator()

class CH21EOS:
    """
    CH21EOS -- EOS class for Chabrier Debras 2021 EOS tables

    This class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions
    to interpolate the Chabrier Debras 2021 EOS tables.

    Parameters
    ----------
    tablePath : str
        Path to the Chabrier Debras 2021 EOS table
    
    Attributes
    ----------
    TRange : tuple
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
    rhoRange : tuple
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table

    Methods
    -------
    pressure(logT, logRho)
        Interpolates the pressure at the given temperature and density
    energy(logT, logRho)
        Interpolates the energy at the given temperature and density

    Example Usage
    -------------
    >>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
    >>> eos = CH21EOS("path/to/eos/table")
    >>> pressure = eos.pressure(7.0, -2.0)
    >>> energy = eos.energy(7.0, -2.0)
    >>> print(pressure, energy)
    """
    def __init__(self, tablePath):
        """
        Initialize the CH21EOS class with the given table path

        Parameters
        ----------
        tablePath : str
            Path to the Chabrier Debras 2021 EOS table
        """
        self._logger = logging.getLogger("CoolDwarf.EOS.ChabrierDebras2021.EOS.CH21EOS")
        self._logger.info(f"Chabrier Debras 2021 EOS intialized with table {tablePath}")
        self._tablePath = tablePath
        self.parse_table()

    def parse_table(self):
        """
        Parse the Chabrier Debras 2021 EOS table and store the data in the class. Parsing is done using regular
        expressions to extract the data from the table file. Columns are named according to the table format, and the
        data is stored in a pandas DataFrame. The temperature and density values are extracted and stored in separate
        arrays, and the pressure and energy values are interpolated using RegularGridInterpolator.

        Column Names (in order):
        - logT
        - logP
        - logRho
        - logU
        - logS
        - dlrho/dlT_P
        - dlrho/dlP_T
        - dlS/dlT_P
        - dlS/dlP_T

        The EOS table is stored as a 3D array, with the first dimension corresponding to the temperature, the second
        dimension corresponding to the density, and the third dimension corresponding to the columns.

        The temperature and density arrays are stored as 1D arrays.
        """
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
        self._temps = xp.array(self._temps)
        self._EOSTabs = xp.array(self._EOSTabs)
        self._rhos = self._EOSTabs[0, :, 2]
        self._logger.info(f"Found EOS Tab array of shape {self._EOSTabs.shape}")
        self._logger.debug(f"Found Temps array of shape {self._temps.shape}")
        self._logger.debug(f"Found Density array of shape {self._rhos.shape}")

        self._forward_pressure = RegularGridInterpolator((self._temps, self._rhos), self._EOSTabs[:, :, 1])
        self._forward_energy = RegularGridInterpolator((self._temps, self._rhos), self._EOSTabs[:, :, 3])

    def check_forward_params(self, logT, logRho):
        """
        Check if the given temperature and density are within the bounds of the EOS table. If the values are out of
        bounds, a ValueError is raised.

        Parameters
        ----------
        logT : float
            Log10 of the temperature in K
        logRho : float
            Log10 of the density in g/cm^3
        
        Raises
        ------
        ValueError
            If the temperature or density is out of bounds
        """
        mask = (self._temps.min() <= logT) & (logT <= self._temps.max())
        if not mask.all():
            self._logger.error(f"Temperature (log10T) is not in bounds of EOS table -- {logT} ∉ ({self._temps.min():0.3f}, {self._temps.max():0.3f})")
            raise ValueError(f"Temperature (log10T) is not in bounds of EOS table -- {logT} ∉ ({self._temps.min():0.3f}, {self._temps.max():0.3f})")
        mask = (self._rhos.min() <= logRho) & (logRho <= self._rhos.max())
        if not mask.all():
            self._logger.error((f"Density (log10Rho) is not in bounds of EOS table -- {logRho} ∉ ({self._rhos.min():0.3f}, {self._rhos.max():0.3f})"))
            raise ValueError(f"Density (log10Rho) is not in bounds of EOS table -- {logRho} ∉ ({self._rhos.min():0.3f}, {self._rhos.max():0.3f})")

    def pressure(self, logT, logRho):
        """
        Find the pressure at the given temperature and density.

        Parameters
        ----------
        logT : float
            Log10 of the temperature in K

        logRho : float
            Log10 of the density in g/cm^3
        """
        logT = xp.array(logT)
        logRho = xp.array(logRho)
        self.check_forward_params(logT, logRho)
        return 10**self._forward_pressure((logT, logRho))

    def energy(self, logT, logRho):
        """
        Find the energy at the given temperature and density.

        Parameters
        ----------
        logT : float
            Log10 of the temperature in K
        logRho : float
            Log10 of the density in g/cm^3
        """
        logT = xp.array(logT)
        logRho = xp.array(logRho)
        self.check_forward_params(logT, logRho)
        return 10**self._forward_energy((logT, logRho))

    def energy_torch(self, logT, logRho):
        """
        Find the energy at the given temperature and density using PyTorch tensors.

        Parameters
        ----------
        logT : torch.Tensor
            Log10 of the temperature in K
        logRho : torch.Tensor
            Log10 of the density in g/cm^3
        """
        if CUPY:
            logT_cp = xp.array(logT.cpu().detach().numpy())
            logRho_cp = xp.array(logRho.cpu().detach().numpy())
        else:
            logT_cp = logT.cpu().detach().numpy()
            logRho_cp = logRho.cpu().detach().numpy()
        self.check_forward_params(logT_cp, logRho_cp)
        energy_cp = self._forward_energy((logT_cp, logRho_cp))
        return torch.tensor(10**energy_cp, device=logT.device)

    @property
    def TRange(self):
        """
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table

        Returns
        -------
        tuple
            Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
        """
        return (self._temps.min(), self._temps.max())

    @property
    def rhoRange(self):
        """
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table

        Returns
        -------
        tuple
            Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table
        """
        return (self._rhos.min(), self._rhos.max())

