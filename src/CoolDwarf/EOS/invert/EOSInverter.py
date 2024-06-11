"""
EOSInverter.py -- Inverter class for EOS tables

This module contains the Inverter class for EOS tables. The class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions to invert the EOS tables.
Because the inversion problem is non-linear, the Inverter class uses the scipy.optimize.minimize function to find the solution.

Further, because EOSs may not be truley invertible, the Inverter class uses a loss function to find the closest solution to the target energy.
over a limited range of temperatures and densities. This is intended to be a range centered around the initial guess for the inversion and
limited in size by some expected maximum deviation from the initial guess.

Dependencies
------------
- numpy
- scipy
- CoolDwarf.err

Example usage
-------------
>>> from CoolDwarf.EOS.invert.EOSInverter import Inverter
>>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
>>> eos = CH21EOS("path/to/eos/table")
>>> inverter = Inverter(eos, TRange, RhoRange)
>>> logTInit, logRhoInit = 7.0, -2.0
>>> newTRange = (6.0, 8.0)
>>> newRhoRange = (-3.0, 0.0)
>>> energy = 1e15
>>> newBounds = (newTRange, newRhoRange)
>>> inverter.set_bounds(newBounds)
>>> logT, logRho = inverter.temperature_density(energy, logTInit, logRhoInit)
"""
from scipy.optimize import minimize
import numpy as np
from typing import Tuple

from CoolDwarf.err import EOSInverterError, EOSBoundsError

class Inverter:
    """
    Inverter -- Inverter class for EOS tables

    This class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions
    to invert the EOS tables. The Inverter class uses the scipy.optimize.minimize function to find the solution to the
    non-linear inversion problem. Because EOSs may not be truley invertible, the Inverter class uses a loss function to
    find the closest solution to the target energy over a limited range of temperatures and densities. This is intended
    to be a range centered around the initial guess for the inversion and limited in size by some expected maximum deviation
    from the initial guess.

    Parameters
    ----------
    EOS : EOS
        EOS object to invert
    TRange : tuple
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
    RhoRange : tuple
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table

    Attributes
    ----------
    EOS : EOS
        EOS object to invert
    _TRange : tuple
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
    _RhoRange : tuple  
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table
    _bounds : tuple
        Tuple containing the TRange and RhoRange
    
    Methods
    -------
    temperature_density(energy, logTInit, logRhoInit)
        Inverts the EOS to find the temperature and density that gives the target energy
    set_bounds(newBounds)
        Sets the bounds for the inversion

    Example Usage
    -------------
    >>> from CoolDwarf.EOS.invert.EOSInverter import Inverter
    >>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
    >>> eos = CH21EOS("path/to/eos/table")
    >>> inverter = Inverter(eos, TRange, RhoRange)
    >>> logTInit, logRhoInit = 7.0, -2.0
    >>> newTRange = (6.0, 8.0)
    >>> newRhoRange = (-3.0, 0.0)
    >>> energy = 1e15
    >>> newBounds = (newTRange, newRhoRange)
    >>> inverter.set_bounds(newBounds)
    >>> logT, logRho = inverter.temperature_density(energy, logTInit, logRhoInit)
    >>> print(logT, logRho)
    """
    def __init__(self, EOS, TRange, RhoRange):
        """
        Initialize the Inverter class

        Parameters
        ----------
        EOS : EOS
            EOS object to invert
        TRange : tuple
            Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
        RhoRange : tuple
            Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table
        """
        self.EOS = EOS
        self._validate_bounds((TRange, RhoRange))
        self._TRange = TRange
        self._RhoRange = RhoRange
        self._bounds = (self._TRange, self._RhoRange)

    def temperature_density(self, energy: float , logTInit: float, logRhoInit: float) -> np.ndarray:
        """
        Inverts the EOS to find the temperature and density that gives the target energy

        Parameters
        ----------
        energy : float
            Target energy to invert the EOS to
        logTInit : float
            Initial guess for the temperature (log10(T)) to start the inversion
        logRhoInit : float
            Initial guess for the density (log10(ρ)) to start the inversion

        Returns
        -------
        np.ndarray
            Array containing the temperature and density that gives the target energy

        Raises
        ------
        EOSInverterError
            If the inversion fails to find a solution
        """
        x0 = [logTInit, logRhoInit]
        r = minimize(self._loss, x0, energy, bounds=self._bounds, method="Nelder-Mead")
        if not r.success:
            raise EOSInverterError(f"No Inversion found for U={energy:0.3f} within (logT, logRho) = {self._bounds}")
        logT = r.x[0]
        logRho = r.x[1]
        return np.array([logT, logRho])

    def _loss(self, x: np.ndarray, target: float) -> float:
        """
        Simple loss function to minimize the difference between the target energy and the EOS energy

        Parameters
        ----------
        x : np.ndarray
            Array containing the temperature and density to evaluate the loss function at
        target : float
            Target energy to invert the EOS to
        
        Returns
        -------
        float
            Loss function value
        """
        logT = x[0]
        logRho = x[1]
        l = np.abs(self.EOS.energy(logT, logRho) - target)
        return l

    def set_bounds(self, newBounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        Sets the bounds for the inversion

        Parameters
        ----------
        newBounds : tuple
            Tuple containing the new bounds for the inversion

        Returns
        -------
        bool
            True if the bounds are set successfully, False otherwise

        Example Usage
        -------------
        >>> newTRange = (6.0, 8.0)
        >>> newRhoRange = (-3.0, 0.0)
        >>> newBounds = (newTRange, newRhoRange)
        >>> inverter.set_bounds(newBounds)
        """
        try:
            self._validate_bounds(newBounds)
        except EOSBoundsError as e:
            return False
        self._TRange = newBounds[0]
        self._RhoRange = newBounds[1]
        self._bounds = (self._TRange, self._RhoRange)
        return True
        

    def _validate_bounds(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        """
        Validates the bounds provided to the EOS inverter

        Parameters
        ----------
        bounds : tuple
            Tuple containing the bounds for the inversion

        Raises
        ------
        EOSBoundsError
            If the bounds are invalid
        """
        Tl, Th = bounds[0][0], bounds[0][1]
        Rl, Rh = bounds[1][0], bounds[1][1]

        try:
            assert self.EOS._temps.min() <= Tl <= self.EOS._temps.max(), f"{self.EOS._temps.min()}, {Tl}, {self.EOS._temps.max()}" 
            assert self.EOS._temps.min() <= Th <= self.EOS._temps.max(), f"{self.EOS._temps.min()}, {Th}, {self.EOS._temps.max()}"
            assert Tl < Th, f"{Tl}, {Th}"
            assert self.EOS._rhos.min() <= Rl <= self.EOS._rhos.max(), f"{self.EOS._rhos.min()}, {Rl}, {self.EOS._rhos.max()}"
            assert self.EOS._rhos.min() <= Rh <= self.EOS._rhos.max(), f"{self.EOS._rhos.min()}, {Rh}, {self.EOS._rhos.max()}"
            assert Rl < Rh, f"{Rl}, {Rh}"
        except AssertionError as e:
            raise EOSBoundsError(f"Invalid Bounds provided to EOS inverter: {e}")
