from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.utils.math import spherical_grid_equal_volume
from CoolDwarf.star import VoxelSphere
from CoolDwarf.utils.const import CONST
from CoolDwarf.star import default_tol

from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Union, Dict

xp, CUPY = get_array_module()

class AdiabaticIdealAtmosphere:
    def __init__(
            self,
            structure : VoxelSphere,
            radialResolution : int = 10,
            azimuthalResolution : int = 10,
            altitudinalResolution : int = 10,
            tol : Dict[str, float]  = default_tol(),
            dof : int = 5,
            Omega : float = 0
            ):
        self.si, self.sj, self.sk = structure._get_grid_index(structure.radius, 0, 0)
        self.Rs = structure.R[self.si, self.sj, self.sk]
        self.structure = structure
        self.radialResolution = radialResolution
        self.azimuthalResolution = azimuthalResolution
        self.altitudinalResolution = altitudinalResolution
        self.Omega = Omega
        self._tolerances = tol
        self.dof = dof

        self.rho0 : float = structure.density[self.si, self.sj, self.sk]
        self.To : float = structure.temperature[self.si, self.sj, self.sk]
        self.Po : float = structure.pressure[self.si, self.sj, self.sk]

        self.g : float= structure.gravitational_acceleration[self.si, self.sj, self.sk]
        self.Cv : float = (dof/2) * CONST['R']
        self.Cp : float = self.Cv + CONST['R']
        self._mu = self.structure._effectiveMolarMass * 1.66054e-24
        self.H : float = (CONST["kB"] * self.To)/(self._mu * self.g)
        self.atmoHeight = self.Rs + self.H

        (self.R, self.THETA, self.PHI, self.r, self.theta, self.phi,
        self._volumeGrid, self._volumeError) = spherical_grid_equal_volume(
                self.radialResolution,
                self.azimuthalResolution,
                self.altitudinalResolution,
                self.atmoHeight,
                self._tolerances['volCheck'],
                minR = self.Rs
                )

        self._temperatureGrid = self.temperature_profile(self.R)
        self._densityGrid = self.density_profile(self.R)
        self._pressureGrid = self.pressure_profile(self.R)
        self._energyGrid = (self.dof/2) * ((CONST['kB'] * self._temperatureGrid)/self._mu) * self._densityGrid * self._volumeGrid


    def density_profile(self, r : Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:   
        z = r - self.Rs
        rho = self.rho0 * (1 - z/self.H)**((self.Cp/CONST['R'])-1)
        return rho

    def temperature_profile(self, r : Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        z = r - self.Rs
        T = self.To * (1 - z/self.H)
        return T

    def pressure_profile(self, r : Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        z = r - self.Rs
        radialPressure = self.Po * (1 - z/self.H)**(self.Cp/CONST['R'])
        geostrophicPressure = ((self._densityGrid * self.Omega**2 * self.R)/4) * (1 + xp.cos(2*self.PHI))
        return radialPressure + geostrophicPressure

    def adiabatic_loss(self) -> float:
        return -self.g/self.Cp

    def advective_loss(self):
        # using correction false as I broke out the jacobian into the advection term
        v = self.geostrophic_wind_velocity_field
        grT = self.gradT(corr=False)
        advection = (v[1]/(self.R*xp.cos(self.PHI))) * grT[1] + (v[2]/self.R) * grT[2]
        return advection

    def diffusive_loss(self):
        # using correction false as I broke out the jacobian into the diffusion term
        grT = self.gradT(corr=False)
        lapT = self.gradT(order=2, corr=False)
        alpha = (self.k/(self._densityGrid * self.Cp))
        adjustPhi = xp.cos(self.PHI) * grT[2]
        lapAdjust = xp.gradient(adjustPhi, self.phi, axis=2)
        diffusion = (alpha/self.R**2) * ((1/xp.cos(self.PHI)**2)*lapT[1] + lapAdjust/xp.cos(self.PHI))
        return diffusion

    def radiative_loss(self):
        # TODO Impliment this properly. I.e. find the radiative loss
        return 0

    def timestep_temperature(self, dt : float):
        """
        Calculate the change in temperature over some timestep, dt.
        The change in temperature is given by the following equation:
            .. math::
                \\Delta T = \\lambda_{D} - \\lambda_{A} + \\lambda_{R}

        Where :math:`\\lambda_{D}` is the diffusive loss, :math:`\\lambda_{A}` is the advective loss,
        and :math:`\\lambda_{R}` is the radiative loss. 

        Parameters
        ----------
        dt : float
            The timestep to calculate the change in temperature over.

        Returns
        -------
        deltaT : float
            The change in temperature over the timestep, dt.
        """
        # TODO Also add heating from the surface of the star. This is important probably to keep the atmosphere
        # hot and not cooling down too much
        lambdaD = self.diffusive_loss()
        lambdaA = self.advective_loss()
        lambdaR = self.radiative_loss()
        return (lambdaD - lambdaA + lambdaR) * dt

    @property
    def k(self):
        # TODO Impliment this properly. I.e. find the thermal conducitivity
        return 1

    @property
    def geostrophic_wind_velocity_field(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pressureGradient = self.gradP
        ur = np.zeros_like(self.R)
        utheta = (1/(self._densityGrid * self.f * self.R * xp.sin(self.PHI))) * pressureGradient[2]
        upi = -(1/self._densityGrid * self.f * self.R) * pressureGradient[1]
        return (ur, utheta, upi)

    @property
    def temperature(self) -> NDArray[np.float64]:
        return self._temperatureGrid.copy()

    @property
    def pressure(self) -> NDArray[np.float64]:
        return self._pressureGrid.copy()

    @property
    def density(self) -> NDArray[np.float64]:
        return self._densityGrid.copy()

    def gradT(self, order : int = 1, zThresh : float = 1e-8, corr : bool = True) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Computes the temperature gradients in the radial, azimuthal, and altitudinal directions.

        Parameters
        ----------
        order : int, default=1
            The order of the derivative.
        zThresh : float, default=1e-8
            The threshold value for the temperature gradient. If the absolute
            value of the gradient is less than zThresh, it is set to 0.
        coor : bool, default=True
            If True, the temperature gradient is computed in spherical coordinates.
            If False, the gradient is computed in cartesian coordinates. More formally,
            if true then the gradient is computed as:

            .. math::
                \\nabla T = (\\partial_r T, \\frac{1}{r} \\partial_{\\theta} T, \\frac{1}{r \\sin(\\phi)} \\partial_{\\phi} T)

            If false, the gradient is computed as:

            .. math::
                \\nabla T = (\\partial_x T, \\partial_y T, \\partial_z T)

            A similar pattern is carried over to the laplacian.


        Returns
        -------
            tuple: A tuple of 3D arrays representing the temperature gradients in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (tGradR, tGradTheta, tGradPhi).

        Raises
        ------
        TypeError
            If order is not an integer
        ValueError
            If order is less than 1.
        """
        if not isinstance(order, int):
            raise TypeError(f"Order must be of type integer, not {type(order)}")
        if order != 1 and order != 2:
            raise ValueError(f"Order must be 1 or 2, not {order}")

        tGradR, tGradTheta, tGradPhi = xp.gradient(self._temperatureGrid, self.r, self.theta, self.phi)
        if corr:
            tGradTheta = tGradTheta/(self.R * xp.sin(self.PHI))
            tGradPhi = tGradPhi/(self.R)

        # Find higher order temperature derivitives by recursivley calling gradient
        if order == 2:
            if corr:
                tGradRAdjust = self.R**2 * tGradR
                tGradPhiAdjust = xp.sin(self.PHI) * tGradPhi
            else:
                tGradRAdjust = tGradR
                tGradPhiAdjust = tGradPhi

            tGradR = xp.gradient(tGradRAdjust, self.r, axis=0)
            tGradTheta = xp.gradient(tGradTheta, self.theta, axis=1)
            tGradPhi = xp.gradient(tGradPhiAdjust, self.phi, axis=2)

            if corr:
                tGradR = tGradR/(self.R**2)
                tGradTheta = tGradTheta/(self.R**2 * (xp.sin(self.PHI))**2)
                tGradPhi = tGradPhi/(self.R**2 * xp.sin(self.PHI))

        tGradR[abs(tGradR) < zThresh] = 0
        tGradTheta[abs(tGradTheta) < zThresh] = 0
        tGradPhi[abs(tGradPhi) < zThresh] = 0

        return (tGradR, tGradTheta, tGradPhi)

    @property
    def gradP(self, zThresh : float = 1e-8) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pGradR, pGradTheta, pGradPhi = xp.gradient(self._pressureGrid, self.r, self.theta, self.phi)

        pGradTheta = pGradTheta/(self.R * xp.sin(self.PHI))
        pGradPhi = pGradPhi/(self.R)

        pGradR[abs(pGradR) < zThresh] = 0
        pGradTheta[abs(pGradTheta) < zThresh] = 0
        pGradPhi[abs(pGradPhi) < zThresh] = 0

        return (pGradR, pGradTheta, pGradPhi)

    @property
    def f(self) -> NDArray[np.float64]:
        return 2 * self.Omega * xp.sin(self.PHI)

    def __repr__(self):
        return f"AdiabaticIdealAtmosphere({self.structure}, {self.radialResolution}, {self.azimuthalResolution}, {self.altitudinalResolution}, {self._tolerances}, {self.Omega})"


