"""
EOSInverter.py -- Inverter class for EOS tables

This module contains the Inverter class for EOS tables. The class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions to invert the EOS tables.
Because the inversion problem is non-linear, the Inverter class uses the scipy.optimize.minimize function to find the solution.

Further, because EOSs may not be truley invertible, the Inverter class uses a loss function to find the closest solution to the target energy.
over a limited range of temperatures and densities. This is intended to be a range centered around the initial guess for the inversion and
limited in size by some expected maximum deviation from the initial guess.

Dependencies
------------
- CoolDwarf.utils.misc.backend

Example usage
-------------
>>> from CoolDwarf.EOS.invert.EOSInverter import Inverter
>>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
>>> eos = CH21EOS("path/to/eos/table")
>>> inverter = Inverter(eos)
>>> logTInit, logRhoInit = 7.0, -2.0
>>> energy = 1e15
>>> logT, logRho = inverter.temperature_density(energy, logTInit, logRhoInit)
"""

from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

class Inverter:
    """
    Inverter -- Inverter class for EOS tables

    This class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions
    to invert the EOS tables. The Inverter class uses PyTorch optimizers to find the solution to the non-linear inversion problem.
    Because EOSs may not be truly invertible, the Inverter class uses a loss function to find the closest solution to the target energy
    over a limited range of temperatures and densities. This is intended to be a range centered around the initial guess for the inversion
    and limited in size by some expected maximum deviation from the initial guess.

    Parameters
    ----------
    EOS : EOS
        EOS object to invert

    Attributes
    ----------
    EOS : EOS
        EOS object to invert
    
    Methods
    -------
    temperature_density(energy, logTInit, logRhoInit,f=0.01)
        Inverts the EOS to find the temperature and density that gives the target energy
    """
    def __init__(self, EOS):
        """
        Initialize the Inverter class

        Parameters
        ----------
        EOS : EOS
            EOS object to invert
        """
        self.EOS = EOS
    
    def temperature_density(self, energy: xp.ndarray, temperature: xp.ndarray, density: xp.ndarray, f=0.01) -> xp.ndarray:
        """
        Given the target energy, temperature and density, find the temperature and density that gives the target energy.
        This is dones by makining the assumption that the EOS is linear in the range of temperatures and densities given by the bounds.
        If this is true then a function rho(E) at some constant temperature is well defined.

        We define two functions: rho(E)_{T0} and rho(E)_{T1} as the density as a function of energy at two constant 
        temperatures. These temperatures are taken as some fraction (f) less than the initial temperature guess and
        that same fraction greater than the initial temperature guess. 


        Once these two linear functions have been found we evaluate them at the target energy.
        This gives us the density which results in the target energy at two different constant temperaratures. 
        We can then fit a third linear function rho(T) using these two points to pull out a linear approximation for
        the isoenergy curve over the search domain.

        The question then becomes: where along this isoenergy curve will the grid point move to. Any temperature
        and density on that curve will result in the same final energy. We can think here about some 
        arbitrary path from the initial conditions to a point on the isoenergy curve. Every path has some
        path integral in energy. The most likeley destination is the path which minimizes the path integral.

        Because there is an infinite search space and we do not have an analytic function we need to make some simplifying
        assumptions to actually solve this. We observe that over a limited search domain the equation of state
        is continous and smooth. Further, it monotonically increases with temperature and density. This means that
        the path which minimizes the path integral of energy should be the shortest distance (in temperature, density space)
        between the initial condition and the isoenergy curve.

        Finding this path is then as simple as finding the line perpendicular to the isoenergy curve which pases through 
        the initial condition and then solving for where this line is equal to the isoenercy curve.

        The procedure described above is preformed simultaneously for all grid points and has been formulated
        as a pure matrix problem. Because of this is is very efficient. 

        Notes
        -----
        If you are using an equation of state which is not as well behaved as the Chabrier Debras 2021 EOS
        the assumptions I made here may not work. Notebaly, you will need to check if, within the search domain, the energy varies linearly with density
        at a constant temperature. And if, again within the search domain, if the isoenercy curve is linear
        in density and temperature space. If these are true then the algorithm to find the isoenergy curve should still
        be valid. Secondly, you will need to validate that the shortest path between the initial condition and the isoenergy
        curve is the one which minimizes the energy path integral. If that is also true then this method should reliably find 
        the target energy.

        Parameters
        ----------
        energy : xp.ndarray
            Target energy to invert the EOS to
        temperature : xp.ndarray
            Initial guess for the temperature. This should be in linear space NOT log space.
        density : xp.ndarray
            Initial guess for the density. This should be in linear space NOT in log space.
        f : float
            Fraction of the initial guess to use for the bounds


        Returns
        -------
        xp.ndarray
            New temperature
        xp.ndarray
            New density
        """
        # Define the found bounds of the search domain
        tD = xp.array([temperature - f * temperature, temperature + f * temperature])
        rD = xp.array([density - f*density, density + f * density])

        # Use broadcasting to simulate a meshgrid
        tDb = tD[xp.newaxis, :]
        rDb = rD[:, xp.newaxis]
        TD = tDb + rDb * 0  
        RD = rDb + tDb * 0
        U = self.EOS.energy(xp.log10(TD), xp.log10(RD))

        # Find the slopes and intercepts of both the function rho(E)_{T0,T1} simultaniously 
        Sn = xp.diff(rD.T).T
        Sd = xp.diff(U, axis=1)[0]
        S = Sn/Sd
        B = rD[0] - S * U[0]

        # Solve for what densities coorespond to the target energy on thos two curves
        rp = S * energy + B

        # Connect a line between those two points to draw the isoenergy curve
        Sfn = xp.diff(rp, axis=0)
        Sfd = xp.diff(tD, axis=0)
        Sf = Sfn/Sfd
        Bf = rp[0] - Sf * tD[0]

        # Find the line perpendicular to the isoenergy curve which also passes through the initial condition
        Sp = -1/Sf
        Bp = density - Sp * temperature

        # Solve for the intersection of the two lines
        newT = (Bp - Bf)/(Sf - Sp)
        newR = Sf * newT + Bf
        return newT, newR

