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
        self._pressureGrid = self.pressure_profile(self.R)
        self._densityGrid = self.density_profile(self.R)
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
        P = self.Po * (1 - z/self.H)**(self.Cp/CONST['R'])
        return P

    def adiabatic_loss(self) -> float:
        return -self.g/self.Cp

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

    @property
    def gradT(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Computes the temperature gradients in the radial, azimuthal, and altitudinal directions.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the temperature gradients in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (tGradR, tGradTheta, tGradPhi).
        """
        tGradR, tGradTheta, tGradPhi = xp.gradient(self._temperatureGrid, self.r, self.theta, self.phi)
        tGradR[abs(tGradR) < 1e-8] = 0
        tGradTheta[abs(tGradTheta) < 1e-8] = 0
        tGradPhi[abs(tGradPhi) < 1e-8] = 0
        return (tGradR, tGradTheta, tGradPhi)

    @property
    def gradP(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pGradR, pGradTheta, pGradPhi = xp.gradient(self._pressureGrid, self.r, self.theta, self.phi)
        pGradR[abs(pGradR) < 1e-8] = 0
        pGradTheta[abs(pGradTheta) < 1e-8] = 0
        pGradPhi[abs(pGradPhi) < 1e-8] = 0
        pGradPhi = self._densityGrid * self.Omega ** 2 * self.R * xp.sin(self.PHI) * xp.cos(self.PHI)
        return (pGradR, pGradTheta, pGradPhi)

    @property
    def f(self) -> NDArray[np.float64]:
        return 2 * self.Omega * xp.sin(self.PHI)

    def __repr__(self):
        return f"AdiabaticIdealAtmosphere({self.structure}, {self.radialResolution}, {self.azimuthalResolution}, {self.altitudinalResolution}, {self._tolerances}, {self.Omega})"


