"""
sphere.py

This module contains the VoxelSphere class, which represents a 3D voxelized sphere model for a star. 
The VoxelSphere class provides methods for computing various physical properties of the star, 
including its radius, enclosed mass, energy flux, and more. 

The VoxelSphere class uses an equation of state (EOS) for the star, which can be inverted to compute 
pressure and temperature grids. It also provides methods for updating the energy of the star and 
recomputing its state.

The module imports several utility functions and constants from the CoolDwarf package, 
as well as classes for handling errors related to energy conservation and non-convergence.

All units are in non log space cgs unless otherwise specified.

Dependencies
------------
- numpy
- tqdm
- scipy.interpolate
- CoolDwarf.utils.math
- CoolDwarf.utils.const
- CoolDwarf.utils.format
- CoolDwarf.EOS
- CoolDwarf.model
- CoolDwarf.err

Classes
-------
- VoxelSphere: Represents a 3D voxelized sphere model for a star.

Functions
---------
- default_tol: Returns a dictionary of default numerical tolerances for the cooling model.

Exceptions
----------
- EnergyConservationError: Raised when there is a violation of energy conservation.
- NonConvergenceError: Raised when a computation fails to converge.


Example Usage
-------------
    >>> from CoolDwarf.star import VoxelSphere
    >>> from CoolDwarf.utils.plot import plot_3d_gradients, visualize_scalar_field
    >>> from CoolDwarf.utils import setup_logging
    >>> from CoolDwarf.EOS import get_eos
    >>> from CoolDwarf.opac import KramerOpac
    >>> from CoolDwarf.EOS.invert import Inverter

    >>> import numpy as cp
    >>> import matplotlib.pyplot as plt

    >>> setup_logging(debug=True)

    >>> EOS = get_eos("EOS/TABLEEOS_2021_Trho_Y0292_v1", "CD21")
    >>> opac = KramerOpac(0.7, 0.02)
    >>> sphere = VoxelSphere(8e31, "BrownDwarfMESA/BD_TEST.mod", EOS, opac, radialResolution=50, altitudinalResolition=10, azimuthalResolition=20)
    >>> sphere.evolve(maxTime=3.154e+7, dt=86400)
    >>> print(f"Surface Temp: {sphere.surface_temperature_profile}")
"""
import numpy as cp
import cupy as cp
import torch
import itertools
from cupyx.scipy.interpolate import RegularGridInterpolator
import logging
from tqdm import tqdm

from typing import Tuple

from CoolDwarf.utils.math import partial_derivative_x
from CoolDwarf.utils.const import CONST as CoolDwarfCONST

from CoolDwarf.EOS import Inverter
from CoolDwarf.model import get_model
from CoolDwarf.err import EnergyConservationError, NonConvergenceError, VolumeError, ResolutionError

def default_tol():
    """
    Returns a dictionary of default numerical tolerances for the cooling model.

    Returns
    -------
        dict: A dictionary of default numerical tolerances for the cooling model.
        Keys are 'relax', 'maxEChange', and 'volCheck'.
    """
    return {"relax": 1e-6, "maxEChange": 1e-4, "volCheck": 1e-2}


class VoxelSphere:
    """
    A class to represent a 3D voxelized sphere model for a star.

    Attributes
    ----------
    CONST : dict
        A dictionary of physical constants.
    mass : float
        The mass of the star.
    model : str
        The name of the stellar model to use.
    modelFormat : str
        The format of the stellar model. Default is 'mesa'. May also be 'dsep'.
    EOS : EOS
        The equation of state for the star.
    opac : Opac
        The opacity of the star.
    pressureRegularization : float
        A regularization parameter for the pressure computation.
    radialResolution : int
        The number of radial divisions in the voxelized sphere.
    azimuthalResolution : int
        The number of azimuthal divisions in the voxelized sphere.
    t0 : float
        The initial time of the star.
    X : float
        The hydrogen mass fraction of the star.
    Y : float
        The helium mass fraction of the star.
    Z : float
        The metal mass fraction of the star.
    alpha : float
        The mixing length parameter.
    mindt : float
        The minimum timestep for the star.
    cfl_factor : float
        The Courant-Friedrichs-Lewy factor for the star.
    tol : dict
        A dictionary of numerical tolerances for the cooling model.
        Keys are 'relax' and 'maxEChange'. Default is {'relax': 1e-6, 'maxEChange': 1e-4, 'volCheck': 1e-2}.
        Relax is the relaxation parameter for the energy update. 
        MaxEChange is the maximum fractional change in energy allowed per timestep.
        volCheck is the maximum fractional error in volume allowed.
    
    Raises
    ------
    EnergyConservationError
        If there is a violation of energy conservation.
    NonConvergenceError
        If a computation fails to converge.
    """
    CONST = CoolDwarfCONST

    def __init__(
            self,
            mass,
            model,
            EOS,
            opac,
            pressureRegularization=1e-5,
            radialResolution=10,
            azimuthalResolition=10,
            altitudinalResolition=10,
            t0=0,
            X=0.75,
            Y=0.25,
            Z=0,
            tol=default_tol(),
            modelFormat='mesa',
            alpha=1.901,
            mindt=0.1,
            cfl_factor = 0.5,
            ):
        """
        Constructs a VoxelSphere object with the specified parameters.
        The initial state of the star is computed based on the given stellar model and 
        this will inform the initial temperature and density grids. The temperature
        and density values from the stellar model are interpolated to over the radius
        grid. The pressure and energy grids are then computed using the EOS.
        An inverted EOS is initialized to allow for the computation of temperature and
        density grids from the energy grid.
        
        Parameters
        ----------
        mass : float
            The mass of the star.
        model : str
            The name of the stellar model to use.
        EOS : EOS
            The equation of state for the star.
        opac : Opac
            The opacity of the star.
        pressureRegularization : float, optional
            A regularization parameter for the pressure computation. Default is 1e-5.
        radialResolution : int, optional
            The number of radial divisions in the voxelized sphere. Default is 10.
        azimuthalResolition : int, optional
            The number of azimuthal divisions in the voxelized sphere. Default is 10.
        altitudinalResolition : int, optional
            The number of altitudinal divisions in the voxelized sphere. Default is 10.
        t0 : float, optional
            The initial time of the star. Default is 0.
        X : float, optional
            The hydrogen mass fraction of the star. Default is 0.75.
        Y : float, optional
            The helium mass fraction of the star. Default is 0.25.
        Z : float, optional
            The metal mass fraction of the star. Default is 0.
        tol : dict, optional
            A dictionary of numerical tolerances for the cooling model. Default is {'relax': 1e-6, 'maxEChange': 1e-4}.
            Relax is the relaxation parameter for the energy update. 
            MaxEChange is the maximum fractional change in energy allowed per timestep.
        modelFormat : str, optional
            The format of the stellar model. Default is 'mesa'. May also be 'dsep'.
        alpha : float, optional
            The mixing length parameter. Default is 1.901.
        mindt : float, optional
            The minimum timestep for the star. Default is 0.1.
        cfl_factor : float, optional
            The Courant-Friedrichs-Lewy factor for the star. Default is 0.5.
        """
        self._logger = logging.getLogger("CoolDwarf.star.sphere.VoxelSphere")

        self._logger.info(f"VoxelSphere Initizlized with mass: {mass} from {model} (X, Y, Z) = ({X}, {Y}, {Z}), alpha={alpha}")
        self._logger.info(f"VoxelSphere resolution: r={radialResolution}, theta={azimuthalResolition}, phi={altitudinalResolition}")

        self._model_path = model
        self._mass = mass
        self.alpha = alpha
        self.mindt = mindt
        self.cfl_factor = cfl_factor

        self._torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Using device for torch acceleration: {self._torch_device}")

        self.radialResolution = radialResolution
        self.azimuthalResolition = azimuthalResolition
        self.altitudinalResolition = altitudinalResolition

        self.epsilonH = pressureRegularization

        self._t = t0
        self._X, self._Y, self._Z = X, Y, Z
        self._effectiveMolarMass = self._X * self.CONST['mH'] + self._Y * self.CONST['mHe']
        self._opac = opac

        self._tolerances = tol

        self._1D_structure = get_model(self._model_path, modelFormat)
        self._radius = cp.exp(self._1D_structure.lnR.values.max())
        self._densityf = RegularGridInterpolator(
                cp.exp(cp.array([self._1D_structure.lnR.values])),
                cp.exp(cp.array(self._1D_structure.lnd.values)),
                bounds_error=False,
                fill_value=cp.exp(self._1D_structure.lnd.values.max())
                )
        self._temperaturef = RegularGridInterpolator(
                cp.exp(cp.array([self._1D_structure.lnR.values])),
                cp.exp(cp.array(self._1D_structure.lnT.values)),
                bounds_error=False,
                fill_value=cp.exp(self._1D_structure.lnT.values.max())
                )

        self._eos = EOS
        self._ieos = Inverter(self._eos)

        self._create_voxel_sphere()

        self._evolutionarySteps = 0
        self._t = 0

    @property
    def enclosed_mass(self):
        """
        Computes the enclosed mass of the star as a function of radius.

        Returns
        -------
            interp1d: A 1D interpolation function for the enclosed mass.
        """
        enclosedMass = list()
        radii = cp.linspace(0, cp.exp(self._1D_structure.lnR.values.max()))
        integralMass = cp.trapz(self._densityf(radii), radii)
        for r in radii:
            rs = cp.linspace(0, r)
            em = cp.trapz(self._densityf(rs) * (self._mass/integralMass), rs)
            enclosedMass.append(em)
        enclosedMass = cp.array(enclosedMass)
        return RegularGridInterpolator((radii,), enclosedMass)
        
    @property
    def radius(self):
        """
        Returns the radius of the star.

        Returns
        -------
            float: The radius of the star.
        """
        return self._radius

    def spherical_grid_equal_volume(self, numRadial, numTheta, numPhi, radius):
        """
        Generate points within a sphere with equal volume using a stratified sampling approach.
        Returns radius, theta, phi as meshgrids and volume elements for each point.

        Parameters
        ----------
        numRadial : int
            Number of radial segments.
        numTheta : int
            Number of azimuthal segments.
        numPhi : int
            Number of altitudinal segments.
        radius : float
            Radius of the sphere.
        
        Returns
        -------
            tuple: A tuple of meshgrids for the radial, azimuthal, and altitudinal positions, and the volume elements.

        Raises
        ------
        VolumeError
            If the volume error is greater than the tolerance.

        Example Usage
        -------------
        >>> r, theta, phi, volume = spherical_grid_equal_volume(10, 10, 10, 1)
        """
        rEdges = cp.linspace(0, radius, numRadial + 1)
        r = (rEdges[:-1] + rEdges[1:]) / 2 
        dr = cp.diff(rEdges) 

        thetaEdges = cp.linspace(0, 2 * cp.pi, numTheta + 1)
        theta = (thetaEdges[:-1] + thetaEdges[1:]) / 2 
        dtheta = cp.diff(thetaEdges) 

        phiEdges = cp.linspace(0, cp.pi, numPhi + 1)
        phi = (phiEdges[:-1] + phiEdges[1:]) / 2 
        dphi = cp.diff(phiEdges)  

        R, THETA, PHI = cp.meshgrid(r, theta, phi, indexing='ij')
        dR, dTHETA, dPHI = cp.meshgrid(dr, dtheta, dphi, indexing='ij')

        volumeElements = (R ** 2) * cp.sin(PHI) * dR * dTHETA * dPHI

        discreteVolume = cp.sum(volumeElements)
        trueVolume = 4/3 * cp.pi * radius**3
        
        # Check if the fractional error in the volume difference is within tol['volCheck']
        if abs(discreteVolume - trueVolume) / trueVolume > self._tolerances['volCheck']:
            raise VolumeError(f"Volume error is greater than tolerance: {abs(discreteVolume - trueVolume) / trueVolume}")
        self._logger.info(f"Volume error: {abs(discreteVolume - trueVolume) / trueVolume} is within tolerance ({self._tolerances['volCheck']})")

        return R, THETA, PHI, r, theta, phi, volumeElements

    def _create_voxel_sphere(self):
        """
        Creates the voxelized sphere model for the star.
        The radial, azimuthal, and altitudinal grids are created based on the specified resolutions.
        The temperature, density, volume, and differential mass grids are computed based on the radial grid.
        The mass grid is computed based on the enclosed mass.
        The pressure and energy grids are computed using the EOS.
        """
        self.R, self.THETA, self.PHI, self.r, self.theta, self.phi, self._volumneGrid = self.spherical_grid_equal_volume(self.radialResolution, self.azimuthalResolition, self.altitudinalResolition, self.radius)

        if self.r.size <= 2:
            raise ResolutionError("Minimum of 3 radial points (radialResolution) required")
        if self.theta.size < 2 or self.phi.size < 2:
            raise ResolutionError("Minimum of 2 angular points (azimuthalResolution, altitudinalResolution) required")
        self._dr = self.r[1] - self.r[0]

        if self.theta.size == 1:
            self._dtheta = 2*cp.pi
        else:
            self._dtheta = self.theta[1] - self.theta[0]
        if self.phi.size == 1:
            self._dphi = cp.pi
        else:
            self._dphi = self.phi[1] - self.phi[0]

        self._temperatureGrid = self._temperaturef(self.R.flatten()).reshape(self.R.shape)
        self._densityGrid = self._densityf(self.R.flatten()).reshape(self.R.shape)
        self._differentialMassGrid = self._volumneGrid * self._densityGrid


        self._massGrid = self.enclosed_mass(self.R.flatten()).reshape(self.R.shape)
        self._forward_EOS()

    def _forward_EOS(self):
        """
        Computes the pressure and energy grids using the EOS.
        """
        logT = cp.log10(self._temperatureGrid)
        logRho = cp.log10(self._densityGrid)
        self._pressureGrid = 1e10 * self._eos.pressure(logT, logRho)
        self._energyGrid = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(logT, logRho))

    def _make_TD_search_grid(self, f: float = 0.01) -> Tuple[Tuple[cp.ndarray, cp.ndarray], Tuple[cp.ndarray, cp.ndarray]]:
        """
        Builds a limited 2D domain for the temperature and density grids for the EOS inversion.
        This is used to limit the search space for the EOS inversion and to make the
        inverted EOS one-to-one.

        Returns
        -------
            tuple: A tuple of 2D arrays representing the limited search space for the temperature and density grids.
            this is in the form of ((lowerTRange, upperTRange), (lowerDRange, upperDRange)).
        """
        fT = f * self._temperatureGrid
        fD = f * self._densityGrid
        lowerTRange = cp.log10(self._temperatureGrid - fT)
        upperTRange = cp.log10(self._temperatureGrid + fT)
        lowerDRange = cp.log10(self._densityGrid - fD)
        upperDRange = cp.log10(self._densityGrid + fD)
        return ((lowerTRange, upperTRange), (lowerDRange, upperDRange))

    def _reverse_EOS(self, f: float = 0.01, pbar: bool = False):
        """
        Computes the temperature and density grids from the energy grid using the inverted EOS.
        The first step here is for the energy grid (which is in the form of internal energy at each
        grid point) to be converted to specific internal energy (internal energy per unit mass).

        The temperature and density grids are then iterated over, and the inverted EOS is used to
        compute the temperature and density at each grid point. The temperature and density grids are
        then updated with the computed values.

        At each grid point, the temperature and density are computed by inverting the EOS using the
        specific internal energy, temperature, and density at that grid point. The bounds for the
        inversion are set based on the temperature and density grids at that point.

        Finally, the pressure grid is updated using the new temperature and density grids.

        Parameters
        ----------
        f : float, optional
            A factor to limit the search space for the EOS inversion. Default is 0.01.
        pbar : bool, optional
            A flag to show a progress bar for the inversion. Default is False.
        """
        specificInternalEnergy = (1000 * self._energyGrid)/(1e13 * self._differentialMassGrid)
        initT = torch.from_numpy(cp.log10(self._temperatureGrid).get())
        initRho = torch.from_numpy(cp.log10(self._densityGrid).get())
        energy = torch.from_numpy(specificInternalEnergy.get())
        tRange, dRange = self._make_TD_search_grid(f=f)
        self._ieos.set_bounds(tRange, dRange)
        self._ieos.temperature_density(energy, initT, initRho)

        self._pressureGrid = 1e10 * self._eos.pressure(cp.log10(self._temperatureGrid), cp.log10(self._densityGrid))

    def Cp(self, delta_t: float = 1e-5):
        """
        Computes the specific heat capacity of the star at constant pressure.

        Parameters
        ----------
        delta_t : float, optional
            A small change in temperature for computing the specific heat capacity. Default is 1e-5.

        Returns
        -------
            cp.ndarray: The specific heat capacity of the star at constant pressure.
        """
        u1 = self._energyGrid
        u2 = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(cp.log10(self._temperatureGrid + delta_t), cp.log10(self._densityGrid)))
        
        cpv = (u2 - u1) / delta_t
        cp_specific = cpv / self._effectiveMolarMass
        cp_specific[cp_specific == 0] = cp.inf
        return cp_specific

    @property
    def gradT(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the temperature gradients in the radial, azimuthal, and altitudinal directions.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the temperature gradients in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (tGradR, tGradTheta, tGradPhi).
        """
        tGradR, tGradTheta, tGradPhi = cp.gradient(self._temperatureGrid, self.r, self.theta, self.phi)
        tGradR[abs(tGradR) < 1e-8] = 0
        tGradTheta[abs(tGradTheta) < 1e-8] = 0
        tGradPhi[abs(tGradPhi) < 1e-8] = 0
        return (tGradR, tGradTheta, tGradPhi)

    @property
    def gradRadEr(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the radiative energy gradient.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the radiative energy gradient in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (delErR, delErTheta, delErPhi).
        """
        tGradR, tGradTheta, tGradPhi = self.gradT
        c0 = 4*self.CONST['a'] * self._temperatureGrid**3
        delErR = c0*tGradR
        delErTheta = c0*tGradTheta
        delErPhi = c0*tGradPhi
        return (delErR, delErTheta, delErPhi)

    @property
    def radiative_energy_flux(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the radiative energy flux in the radial, azimuthal, and altitudinal directions.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the radiative energy flux in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (fluxRadR, fluxRadTheta, fluxRadPhi).
        """
        opacity = self._opac.kappa(self._temperatureGrid, self._densityGrid)
        c0 = -(self.CONST['c']/(3*opacity*self._densityGrid))
        energyGradient = self.gradRadEr
        fluxGradR = c0 * energyGradient[0]
        fluxGradTheta = c0 * energyGradient[1]
        fluxGradPhi = c0 * energyGradient[2]
        return (fluxGradR, fluxGradTheta, fluxGradPhi)

    @property
    def convective_energy_flux(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the convective energy flux in the radial, azimuthal, and altitudinal directions.
        We take a mixing length theory approach to compute the convective energy flux.

        The convective enertgy flux for some coordinate direction is given by:
        Fconv = (1/2) * rho * Cp * v * (TGrad - ad)
        where rho is the density, Cp is the specific heat capacity, v is the convective velocity along that coordinate axis,
        TGrad is the temperature gradient along that coordinte axis, and ad is the adiabatic gradient.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the convective energy flux in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (FconvR, FconvTheta, FconvPhi).
        """

        tGradR, tGradTheta, tGradPhi = self.gradT
        vR, vTheta, vPhi = self.convective_velocity
        ad = self._adiabatic_grad
        cpv = self.Cp()
        density = self._densityGrid

        FradR = (1/2) * density * cpv * vR  * (tGradR - ad)
        FradTheta = (1/2) * density * cpv * vTheta  * (tGradTheta - ad)
        FradPhi = (1/2) * density * cpv * vPhi  * (tGradPhi - ad)
        return (FradR, FradTheta, FradPhi)

    @property
    def convective_overturn_timescale(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the convective overturn timescale in the radial, azimuthal, and altitudinal directions.
        The convective overturn timescale is given by the mixing length divided by the convective velocity.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the convective overturn timescale in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (tauR, tauTheta, tauPhi).
        """

        vR, vTheta, vPhi = self.convective_velocity
        mixingLength = self.mixing_length

        # Deal with singularity at r==0
        vR[vR == 0] = cp.nan
        vTheta[vTheta == 0] = cp.nan
        vPhi[vPhi == 0] = cp.nan
        tauR = mixingLength / vR
        tauTheta = mixingLength / vTheta
        tauPhi = mixingLength / vPhi

        # Correct the output post singularity to reflect limit
        tauR[cp.isnan(vR)] = cp.inf
        tauTheta[cp.isnan(vTheta)] = cp.inf
        tauPhi[cp.isnan(vPhi)] = cp.inf
        return (tauR, tauTheta, tauPhi)

    @property
    def mixing_length(self) -> cp.ndarray:
        """
        Computes the mixing length for the star.

        Returns
        -------
            cp.ndarray: The mixing length for the star.
        """
        return self.alpha*self.pressure_scale_height

    @property
    def pressure_scale_height(self) -> cp.ndarray:
        """
        Computes the pressure scale height for the star.

        Returns
        -------
            cp.ndarray: The pressure scale height for the star.
        """

        g = self.gravitational_acceleration
        H = self._pressureGrid/(self._densityGrid * g + self.epsilonH)
        return H

    @property
    def convective_velocity(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Computes the convective velocity in the radial, azimuthal, and altitudinal directions.
        The convective velocity is given by the mixing length divided by two times the square root of the product of the
        gravitational acceleration and the difference between the adiabatic gradient and the temperature gradient.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the convective velocity in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of (vR, vTheta, vPhi).
        """
        ad = self._adiabatic_grad
        vc = lambda tg: (self.mixing_length/2) * cp.sqrt(self.gravitational_acceleration * (ad-tg)/self._temperatureGrid)
        tGradR, tGradTheta, tGradPhi = self.gradT
        vR = vc(tGradR)
        # vTheta = vc(tGradTheta)
        # vPhi = vc(tGradPhi)
        vTheta = cp.zeros_like(vR)
        vPhi = cp.zeros_like(vR)

        return vR, vTheta, vPhi

    @property
    def gravitational_acceleration(self) -> cp.ndarray:
        """
        Computes the gravitational acceleration for the star. If the mass grid is zero at a given grid point,
        the gravitational acceleration is set to infinity to deal with the singularity at r=0.

        Returns
        -------
            cp.ndarray: The gravitational acceleration for the star.
        """
        rUse = self.R.copy()
        rUse[self._massGrid == 0] = cp.inf # deal with the singularity at r=0
        return (self.CONST['G'] * self._massGrid)/(rUse**2)

    @property
    def _adiabatic_grad(self) -> cp.ndarray:
        """
        Computes the adiabatic gradient for the star.

        Returns
        -------
            cp.ndarray: The adiabatic gradient for the star.
        """

        self._delad = (self._pressureGrid) / (self._densityGrid * self.Cp())
        return self._delad

    @property
    def energy_flux(self) -> Tuple[Tuple[cp.ndarray, cp.ndarray, cp.ndarray], Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        """
        Computes the energy flux in the radial, azimuthal, and altitudinal directions.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the energy flux in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of ((fluxR, fluxTheta, fluxPhi), (fluxR, fluxTheta, fluxPhi)).
        """
        convectiveFlux = self.convective_energy_flux
        radiativeFlux = self.radiative_energy_flux
        return (convectiveFlux, radiativeFlux)

    @property
    def flux_divergence(self) -> Tuple[Tuple[cp.ndarray, cp.ndarray, cp.ndarray], Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        """
        Computes the divergence of the energy flux in the radial, azimuthal, and altitudinal directions.

        Returns
        -------
            tuple: A tuple of 3D arrays representing the divergence of the energy flux in the radial, azimuthal, and altitudinal directions.
            The arrays are in the form of ((delFConvR, delFConvTheta, delFConvPhi), (delFRadR, delFRadTheta, delFRadPhi)).
        """

        flux = self.energy_flux
        dR = self.r[1] - self.r[0]
        dTheta = self.theta[1] - self.theta[0]
        dPhi = self.phi[1] - self.phi[0]
        
        delFConvR = partial_derivative_x(flux[0][0], dR)
        delFConvTheta = partial_derivative_x(flux[0][1], dTheta)
        delFConvPhi = partial_derivative_x(flux[0][2], dPhi)
        delFRadR = partial_derivative_x(flux[1][0], dR)
        delFRadTheta = partial_derivative_x(flux[1][1], dTheta)
        delFRadPhi = partial_derivative_x(flux[1][2], dPhi)
        return ((delFConvR, delFConvTheta, delFConvPhi), (delFRadR, delFRadTheta, delFRadPhi))

    @property
    def dEdt(self) -> cp.ndarray:
        """
        Computes the time derivative of the energy for the star.

        Returns
        -------
            cp.ndarray: The time derivative of the energy for the star.
        """

        fluxDivergence = self.flux_divergence
        dEConvdt = -fluxDivergence[0][0] - fluxDivergence[0][1] - fluxDivergence[0][2]
        dERadt = -fluxDivergence[1][0] - fluxDivergence[1][1] - fluxDivergence[1][2]
        DEDT = dEConvdt + dERadt
        # DEDT = dERadt
        return DEDT

    def _update_energy(self, dt: float):
        """
        Updates the energy of the star based on the time derivative of the energy and the timestep.

        Parameters
        ----------
        dt : float
            The timestep for the update.
        """
        dE = - self.dEdt * dt
        self._logger.info(f"Energy changing by an average of {cp.mean(dE)}, {cp.mean(dE/self._energyGrid)*100}%")
        self._energyGrid += dE

    @property
    def cfl_dt(self)-> float:
        """
        Computes the timestep based on the Courant-Friedrichs-Lewy (CFL) condition.

        Returns
        -------
            float: The timestep based on the CFL condition.
        """
        vr, vtheta, vphi = self.convective_velocity
        max_velocity = max(cp.max(vr), cp.max(vtheta), cp.max(vphi))
        cfl_dt = self.cfl_factor * self._dr / max_velocity
        return cfl_dt

    def timestep(self, userdt : float = cp.inf, pbar: bool = False) -> float:
        """
        Computes a timestep for the star based on the CFL condition or the user-specified timestep.
        The actual timestep used is the minimum of the CFL timestep and the user-specified timestep, 
        and this will be returned. 

        The energy of the star is then updated based on the computed timestep. Following this, the energy
        is used to invert the EOS and update the temperature and density grids. The energy conservation
        is checked, and if the energy change is greater than the maximum energy change tolerance, the timestep
        is halved and the energy, temperature, density, and pressure grids are reset to their initial values.

        When the energy conservation is satisfied, the evolutionary step is incremented, and the time is updated
        based on the the acutal timestep used. The timestep is then returned.

        Parameters
        ----------
        userdt : float, optional
            The user-specified timestep. Default is cp.inf.
        pbar : bool, optional
            A flag to show a progress bar for the inversion. Default is False.

        Returns
        -------
            float: The actual timestep used for the star.

        Raises
        ------
        EnergyConservationError
            If there is a violation of energy conservation.
        NonConvergenceError
            If the model fails to converge after a certain number of timesteps.

        Examples
        --------
        >>> star = VoxelSphere(...)
        >>> star.timestep()
        """
        dt = min(userdt, self.cfl_dt)
        initTempGrid = self._temperatureGrid.copy()
        initDensityGrid = self._densityGrid.copy()
        initPressureGrid = self._pressureGrid.copy()
        initEnergyGrid = self._energyGrid.copy()
        converged = False
        while not converged:
            try:
                self._update_energy(dt)
                self._reverse_EOS(pbar=pbar)
                energyChange = cp.abs(cp.sum(initEnergyGrid - self._energyGrid))/cp.sum(initEnergyGrid)
                if energyChange > self._tolerances['maxEChange']:
                    raise EnergyConservationError(f"Max change in energy reached {energyChange:0.2E}.")
                converged = True
                self._logger.info(f"Convergence reached for model at timestep {self._evolutionarySteps + 1} with {energyChange * 100:0.3E} % energy variation")
            except EnergyConservationError as e:
                self._logger.info(f"Non Convergence in energy ({e}) with a timestep of {dt}, reducing timestep...")
                self._temperatureGrid = initTempGrid.copy()
                self._densityGrid = initDensityGrid.copy()
                self._pressureGrid = initPressureGrid.copy()
                self._energyGrid = initEnergyGrid.copy()
                dt = dt/2
                if dt < self.mindt:
                    raise NonConvergenceError(f"Model Failed to converge after {self._evolutionarySteps} timesteps")
        self._evolutionarySteps += 1
        self._t += dt
        return dt

    def evolve(self, maxTime : float = 3.154e+7, dt : float = 86400, pbar=False):
        """
        Evolves the star over a specified time period using a specified timestep.

        Parameters
        ----------
        maxTime : float, optional
            The maximum time to evolve the star. Default is 3.154e+7.
        dt : float, optional
            The timestep to use for the evolution. Default is 86400.
        pbar : bool, optional
            Display a progress bar for the evolution. Default is False.
        """
        self._logger.info(f"Evolution started with dt: {dt}, maxTime: {maxTime}")
        with tqdm(total=maxTime, disable=not pbar, desc="Evolution") as pbar:
            while self._t < maxTime:
                dt = min(dt, maxTime - self._t)
                useddt = self.timestep(dt)
                self._logger.evolve(
                    f"i: {self._evolutionarySteps:<10} DT: {useddt:<10.2e}(s) AGE: {self._t:<10.2e}(s) ETotal: {cp.sum(self._energyGrid):<20.2e}(erg)"
                )

                pbar.update(useddt)

    @property
    def temperature(self) -> cp.ndarray:
        """
        Returns the temperature grid for the star.

        Returns
        -------
            cp.ndarray: The temperature grid for the star.
        """

        return self._temperatureGrid


    @property
    def surface_temperature_profile(self) -> cp.ndarray:
        """
        Computes the surface temperature profile for the star.

        Returns
        -------
            cp.ndarray: The surface temperature profile for the star.
        """
        surface = self.R == self.R.max()
        surfaceTemperature = self._temperatureGrid[surface]
        return surfaceTemperature
