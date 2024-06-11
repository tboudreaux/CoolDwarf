"""
karmer.py -- Kramer opacity class for CoolDwarf

This module contains the KramerOpac class, which is used to calculate the Kramer opacity for a given temperature and density.

Example usage
-------------
>>> from CoolDwarf.opac.kramer import KramerOpac
>>> X, Z = 0.7, 0.02
>>> opac = KramerOpac(X, Z)
>>> temp, density = 1e7, 1e-2
>>> kappa = opac.kappa(temp, density)
"""
class KramerOpac:
    """
    KramerOpac -- Kramer opacity class for CoolDwarf

    This class is used to calculate the Kramer opacity for a given temperature and density.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction
    Z : float
        Metal mass fraction

    Methods
    -------
    kappa(temp, density)
        Calculates the Kramer opacity at the given temperature and density
    """
    def __init__(self, X: float, Z: float):
        """
        Initialize the KramerOpac class
        
        Parameters
        ----------
        X : float
            Hydrogen mass fraction
        Z : float
            Metal mass fraction
        """
        self.X = X
        self.Z = Z

    def kappa(self, temp: float, density: float) -> float:
        """
        Function to calculate the Kramer opacity at the given temperature and density

        Parameters
        ----------
        temp : float
            Temperature in Kelvin
        density : float
            Density in g/cm^3
        
        Returns
        -------
        kappa : float
            Kramer opacity at the given temperature and density in cm^2/g
        
        Example Usage
        -------------
        >>> from CoolDwarf.opac.kramer import KramerOpac
        >>> X, Z = 0.7, 0.02
        >>> opac = KramerOpac(X, Z)
        >>> temp, density = 1e7, 1e-2
        >>> kappa = opac.kappa(temp, density)
        >>> print(kappa)
        """
        c0 = 4e25
        c1 = (self.Z*(1+self.X))/2
        c2 = density * (temp**(-3.5))
        return c0*c1*c2
