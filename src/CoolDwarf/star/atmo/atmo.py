from CoolDwarf import rad
from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.utils.math import spherical_grid_equal_volume
from CoolDwarf.star import VoxelSphere
from CoolDwarf.utils.const import CONST
from CoolDwarf.star import default_tol
from CoolDwarf.opac import KramerOpac
from CoolDwarf.rad.LTE import attenuate_flux, optical_depth

from CoolDwarf.err import EnergyConservationError, NonConvergenceError, VolumeError, ResolutionError
from CoolDwarf.typing import Arithmetic

from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Union, Dict

import logging

xp, CUPY = get_array_module()

class AdiabaticIdealAtmosphere:
    def __init__(
            self,
            structure : VoxelSphere,
            opac,
            radialResolution : int = 10,
            azimuthalResolution : int = 10,
            altitudinalResolution : int = 10,
            tol : Dict[str, float]  = default_tol(),
            dof : int = 5,
            Omega : float = 0,
            ):
        self._logger = logging.getLogger("CoolDwarf.star.atmo.AdiabaticIdealAtmosphere")
        self.si, self.sj, self.sk = structure._get_grid_index(structure.radius, 0, 0)
        self.Rs = structure.R[self.si, self.sj, self.sk]
        self.structure = structure
        self.radialResolution = radialResolution
        self.azimuthalResolution = azimuthalResolution
        self.altitudinalResolution = altitudinalResolution
        self.Omega = Omega
        self._tolerances = tol
        self.dof = dof
        self._opacity = opac

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
        self._X = self.R * xp.sin(self.THETA) * xp.cos(self.PHI)
        self._Y = self.R * xp.sin(self.THETA) * xp.sin(self.PHI)
        self._Z = self.R * xp.cos(self.THETA)

        self._temperatureGrid = self.temperature_profile(self.R)
        self._densityGrid = self.density_profile(self.R)
        self._pressureGrid = self.pressure_profile(self.R)
        self._energyGrid = (self.dof/2) * ((CONST['kB'] * self._temperatureGrid)/self._mu) * self._densityGrid * self._volumeGrid
        self._logger.info(f"AdiabaticIdealAtmosphere initialized with structure {self.structure}, radial resolution {self.radialResolution}, azimuthal resolution {self.azimuthalResolution}, altitudinal resolution {self.altitudinalResolution}, tolerances {self._tolerances}, and {Omega}")
        self._evolutionarySteps = 0

    def inject_energy(self, theta, phi, r, energy):
        area = xp.pi * r**2
        flux = energy/area
        gamma = np.arccos(1 - ((2 * r**2)/self.atmoHeight**2))
        da = gamma/2
        _, itu, _ = self._get_grid_index(self.atmoHeight, theta + da, phi)
        _, itd, _ = self._get_grid_index(self.atmoHeight, theta - da, phi)
        _, _, jpu = self._get_grid_index(self.atmoHeight, theta, phi + da)
        _, _, jpd = self._get_grid_index(self.atmoHeight, theta, phi - da)


        if itu == itd:
            ibounds = (itu, itu+1)
        else:
            ibounds = sorted((itu, itd))
        if jpu == jpd:
            jbounds = (jpu, jpu+1)
        else:
            jbounds = sorted((jpu, jpd))




        temperatureColumn  = self._temperatureGrid[:, ibounds[0]:ibounds[1], jbounds[0]:jbounds[1]]
        densityColumn = self._densityGrid[:, ibounds[0]:ibounds[1], jbounds[0]:jbounds[1]]
        print(temperatureColumn.min(), temperatureColumn.max())
        print(densityColumn.min(), densityColumn.max())
        opacityColumn = self._opacity.kappa(temperatureColumn, densityColumn)
        print(opacityColumn)
        ds = np.gradient(self.R[:, itu, jpu]).reshape(temperatureColumn.shape[0], 1, 1)
        dsReShape = np.ones_like(opacityColumn)
        dsReShape *= ds
        tau = np.zeros_like(opacityColumn)
        for radiusID, deltaRadius in enumerate(ds):
            tau[radiusID] = optical_depth(opacityColumn[:radiusID], densityColumn[:radiusID], dsReShape[:radiusID])
        
        print(tau)
        attenuatedFlux = attenuate_flux(flux, tau)
        print(attenuatedFlux)



    def _get_grid_index(self, r, theta, phi):
        """
        Computes the grid index for a specified coordinate.
        Parameters
        ----------
        r : float
            The radial coordinate of the grid point.
        theta : float
            The azimuthal coordinate of the grid point.
        phi : float
            The altitudinal coordinate of the grid point.
        Returns
        -------
            tuple: A tuple of the grid indices for the specified coordinate.
        """
        i = xp.abs(self.r - r).argmin()
        j = xp.abs(self.theta - theta).argmin()
        k = xp.abs(self.phi - phi).argmin()
        return i, j, k


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
        self._logger.debug("Radiative loss not implimented yet but called. Returning 0 (No Cooling or heating)")
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
        self._logger.debug(f"Timestep atmospheric temperature called with lambdaD: {lambdaD.mean()}, lambdaA: {lambdaA.mean()}, lambdaR: {lambdaR}")
        return (lambdaD - lambdaA + lambdaR) * dt

    def timestep(self, dt : float):
        dT = self.timestep_temperature(dt)
        dU = self.Cv * dT
        fractionalEnergyChange = dU/self._energyGrid
        if np.any(fractionalEnergyChange > self._tolerances["maxEChange"]):
            raise EnergyConservationError(f"Energy change is greater than the maximum energy change tolerance. Energy change: {fractionalEnergyChange.mean()}, Tolerance: {self._tolerances['maxEChange']}")
        pTempSTD = xp.std(self._temperatureGrid)
        self._temperatureGrid += dT
        self._logger.debug(f"Atmospheric Temperature standard deviation before timestep (before, after): {xp.std(self._temperatureGrid)}, {pTempSTD}")
        initialTotalEnergy = xp.sum(self._energyGrid)
        self._energyGrid += dU
        finalTotalEnergy = xp.sum(self._energyGrid)
        fractionalEnergyChange = (finalTotalEnergy - initialTotalEnergy)/initialTotalEnergy
        self._logger.debug(f"Initial total energy: {initialTotalEnergy}, Final total energy: {finalTotalEnergy}. Fractional Change: {fractionalEnergyChange}")
        if abs(fractionalEnergyChange) > self._tolerances["maxEChange"]:
            self._logger.error(f"Energy change is greater than the maximum energy change tolerance. Energy change: {fractionalEnergyChange}, Tolerance: {self._tolerances['maxEChange']}")
            raise EnergyConservationError(f"Energy change is greater than the maximum energy change tolerance. Energy change: {fractionalEnergyChange}, Tolerance: {self._tolerances['maxEChange']}")
        self._evolutionarySteps += 1
        return dt

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
    def lapT(self, corr: bool = True) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Find the laplacian of the temperature field. This is
        simply a wrapper around 2nd order gradient

        Parameters
        ----------
        corr : bool, default=True
            If True, the laplacian is computed in spherical coordinates.
            If False, the laplacian is computed in cartesian coordinates.

        Returns
        -------
        lapT : NDArray[np.float64]
            The laplacian of the temperature field.
        """

        lapR, lapTheta, lapPhi = self.gradT(order=2, corr=corr)
        return (lapR, lapTheta, lapPhi)

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


