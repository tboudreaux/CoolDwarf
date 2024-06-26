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
- torch
- cupy
- pandas
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
- VolumeError: Raised when the volume error is greater than the tolerance.
- ResolutionError: Raised when the resolution is insufficient.


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
from scipy.interpolate import RegularGridInterpolator
import torch
from tqdm import tqdm
import pandas as pd

import os
from typing import Tuple
import logging

from CoolDwarf.utils.math import partial_derivative_x
from CoolDwarf.utils.const import CONST as CoolDwarfCONST

from CoolDwarf.EOS import Inverter
from CoolDwarf.model import get_model
from CoolDwarf.err import EnergyConservationError, NonConvergenceError, VolumeError, ResolutionError
from CoolDwarf.err import EOSInverterError
from CoolDwarf.utils.output import binmod
from CoolDwarf.utils.misc.backend import get_array_module, get_interpolator
from CoolDwarf.utils.math import spherical_grid_equal_volume

xp, CUPY = get_array_module()
RegularGridInterpolator = get_interpolator()

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
            imodelOut=False,
            imodelOutCadence=1000,
            imodelOutCadenceUnit='s',
            fmodelOut=True,
            outputDir = "."
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
        imodelOut : bool, optional
            A flag to output the model at each timestep. Default is False.
        imodelOutCadence : int, optional
            The cadence at which to output the model. Default is 1000.
        imodelOutCadenceUnit : str, optional
            The unit of the cadence for outputting the model. Default is 's'
            for seconds. Avalible units are 's' (seconds) and 'i' (iterations)
        fmodelOut : bool, optional
            A flag to output the final model. Default is True.
        outputDir : str, optional
            The output directory for the model files. Default is ".".
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
        self._radius = xp.exp(self._1D_structure.lnR.values.max())
        
        if CUPY:
            radialDomain = xp.exp(xp.array([self._1D_structure.lnR.values]))
            tempDomain = xp.exp(xp.array(self._1D_structure.lnT.values))
            densityDomain = xp.exp(xp.array(self._1D_structure.lnd.values))
        else:
            radialDomain = xp.exp([self._1D_structure.lnR.values])
            tempDomain = xp.exp(self._1D_structure.lnT.values)
            densityDomain = xp.exp(self._1D_structure.lnd.values)

        self._densityf = RegularGridInterpolator(
                radialDomain,
                densityDomain,
                bounds_error=False,
                fill_value=xp.exp(self._1D_structure.lnd.values.max())
                )
        self._temperaturef = RegularGridInterpolator(
                radialDomain,
                tempDomain,
                bounds_error=False,
                fill_value=xp.exp(self._1D_structure.lnT.values.max())
                )

        self._eos = EOS
        self._ieos = Inverter(self._eos)

        self._create_voxel_sphere()

        self._evolutionarySteps = 0
        self._t = 0

        self.imodelOut = imodelOut
        self.imodelOutCadence = imodelOutCadence
        self.imodelOutCadenceUnit = imodelOutCadenceUnit
        self.fmodelOut = fmodelOut

        self._modelOutputController = binmod()
        self._outputDir = outputDir


    def _create_voxel_sphere(self):
        """
        Creates the voxelized sphere model for the star.
        The radial, azimuthal, and altitudinal grids are created based on the specified resolutions.
        The temperature, density, volume, and differential mass grids are computed based on the radial grid.
        The mass grid is computed based on the enclosed mass.
        The pressure and energy grids are computed using the EOS.
        """
        (self.R, self.THETA, self.PHI, self.r, self.theta, self.phi,
        self._volumneGrid, self._volumeError) = spherical_grid_equal_volume(
                self.radialResolution,
                self.azimuthalResolition,
                self.altitudinalResolition,
                self.radius,
                self._tolerances['volCheck']
                )

        if self.r.size <= 2:
            raise ResolutionError("Minimum of 3 radial points (radialResolution) required")
        if self.theta.size < 2 or self.phi.size < 2:
            raise ResolutionError("Minimum of 2 angular points (azimuthalResolution, altitudinalResolution) required")
        self._dr = self.r[1] - self.r[0]

        if self.theta.size == 1:
            self._dtheta = 2*xp.pi
        else:
            self._dtheta = self.theta[1] - self.theta[0]
        if self.phi.size == 1:
            self._dphi = xp.pi
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
        logT = xp.log10(self._temperatureGrid)
        logRho = xp.log10(self._densityGrid)
        self._pressureGrid = 1e10 * self._eos.pressure(logT, logRho)
        self._energyGrid = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(logT, logRho))

    def _make_TD_search_grid(self, f: float = 0.01) -> Tuple[Tuple[xp.ndarray, xp.ndarray], Tuple[xp.ndarray, xp.ndarray]]:
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
        lowerTRange = xp.log10(self._temperatureGrid - fT)
        upperTRange = xp.log10(self._temperatureGrid + fT)
        lowerDRange = xp.log10(self._densityGrid - fD)
        upperDRange = xp.log10(self._densityGrid + fD)
        return ((lowerTRange, upperTRange), (lowerDRange, upperDRange))

    def _reverse_EOS(self, f: float = 0.01):
        """
        Uses the EOS inverter to compute the temperature and density grids from the energy grid.
        See the Inverter class for more details on the inversion process.

        Parameters
        ----------
        f : float, default=0.01
            A factor to limit the search space for the EOS inversion.
        """
        specificInternalEnergy = (1000 * self._energyGrid)/(1e13 * self._differentialMassGrid)
        newT, newR = self._ieos.temperature_density(specificInternalEnergy, self._temperatureGrid, self._densityGrid, f=f)
        self._temperatureGrid = newT
        self._densityGrid = newR
        self._pressureGrid = 1e10 * self._eos.pressure(xp.log10(self._temperatureGrid), xp.log10(self._densityGrid))


    def Cp(self, delta_t: float = 1):
        """
        Computes the specific heat capacity of the star at constant pressure.

        Parameters
        ----------
        delta_t : float, optional
            A small change in temperature for computing the specific heat capacity. Default is 1e-5.

        Returns
        -------
            xp.ndarray: The specific heat capacity of the star at constant pressure.
        """
        u1 = self._energyGrid
        u2 = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(xp.log10(self._temperatureGrid + delta_t), xp.log10(self._densityGrid)))
        
        cpv = (u2 - u1) / delta_t
        cp_specific = cpv / self._effectiveMolarMass
        cp_specific[cp_specific == 0] = xp.inf
        return cp_specific

    @property
    def gradT(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
    def gradRadEr(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
    def radiative_energy_flux(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
    def convective_energy_flux(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
    def convective_overturn_timescale(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
        vR[vR == 0] = xp.nan
        vTheta[vTheta == 0] = xp.nan
        vPhi[vPhi == 0] = xp.nan
        tauR = mixingLength / vR
        tauTheta = mixingLength / vTheta
        tauPhi = mixingLength / vPhi

        # Correct the output post singularity to reflect limit
        tauR[xp.isnan(vR)] = xp.inf
        tauTheta[xp.isnan(vTheta)] = xp.inf
        tauPhi[xp.isnan(vPhi)] = xp.inf
        return (tauR, tauTheta, tauPhi)

    @property
    def mixing_length(self) -> xp.ndarray:
        """
        Computes the mixing length for the star.

        Returns
        -------
            xp.ndarray: The mixing length for the star.
        """
        return self.alpha*self.pressure_scale_height

    @property
    def pressure_scale_height(self) -> xp.ndarray:
        """
        Computes the pressure scale height for the star.

        Returns
        -------
            xp.ndarray: The pressure scale height for the star.
        """

        g = self.gravitational_acceleration
        H = self._pressureGrid/(self._densityGrid * g + self.epsilonH)
        return H

    @property
    def convective_velocity(self) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
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
        vc = lambda tg: (self.mixing_length/2) * xp.sqrt(self.gravitational_acceleration * (ad-tg)/self._temperatureGrid)
        tGradR, tGradTheta, tGradPhi = self.gradT
        vR = vc(tGradR)
        # vTheta = vc(tGradTheta)
        # vPhi = vc(tGradPhi)
        vTheta = xp.zeros_like(vR)
        vPhi = xp.zeros_like(vR)

        return vR, vTheta, vPhi

    @property
    def gravitational_acceleration(self) -> xp.ndarray:
        """
        Computes the gravitational acceleration for the star. If the mass grid is zero at a given grid point,
        the gravitational acceleration is set to infinity to deal with the singularity at r=0.

        Returns
        -------
            xp.ndarray: The gravitational acceleration for the star.
        """
        rUse = self.R.copy()
        rUse[self._massGrid == 0] = xp.inf # deal with the singularity at r=0
        return (self.CONST['G'] * self._massGrid)/(rUse**2)

    @property
    def _adiabatic_grad(self) -> xp.ndarray:
        """
        Computes the adiabatic gradient for the star.

        Returns
        -------
            xp.ndarray: The adiabatic gradient for the star.
        """

        self._delad = (self._pressureGrid) / (self._densityGrid * self.Cp())
        return self._delad

    @property
    def energy_flux(self) -> Tuple[Tuple[xp.ndarray, xp.ndarray, xp.ndarray], Tuple[xp.ndarray, xp.ndarray, xp.ndarray]]:
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
    def flux_divergence(self) -> Tuple[Tuple[xp.ndarray, xp.ndarray, xp.ndarray], Tuple[xp.ndarray, xp.ndarray, xp.ndarray]]:
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
    def dEdt(self) -> xp.ndarray:
        """
        Computes the time derivative of the energy for the star.

        Returns
        -------
            xp.ndarray: The time derivative of the energy for the star.
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
        self._logger.info(f"Energy changing by an average of {xp.mean(dE)}, {xp.mean(dE/self._energyGrid)*100}%")
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
        max_velocity = max(xp.max(vr), xp.max(vtheta), xp.max(vphi))
        cfl_dt = self.cfl_factor * self._dr / max_velocity
        return cfl_dt

    def timestep(self, userdt : float = xp.inf) -> float:
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
            The user-specified timestep. Default is xp.inf.

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
                self._cool_star(dt)
                self._reverse_EOS()
                energyChange = xp.abs(xp.sum(initEnergyGrid - self._energyGrid))/xp.sum(initEnergyGrid)
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

    def evolve(self, maxTime : float = 3.154e+7, dt : float = 86400, pbar=False, callback=lambda s: None, cbc=1, cargs: tuple = ()):
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
        callback : function, optional
            A callback function to call at each timestep. Default is a function that does nothing.
            Function will be called after each timestep. The star object will be passed as an argument.
            The function signature must be callback(star, *args).
        cbc : int, optional
            The cadence at which to call the callback function. 1 meaning every time step. 2 would
            be every other timestep and so on...
        cargs : tuple, optional
            Additional arguments to pass to the callback function.

        Example Usage
        -------------
        >>> from CoolDwarf.star.sphere import VoxelSphere
        >>> from CoolDwarf.utils.io import binmod
        >>> star = VoxelSphere(...)
        >>> star.evolve()
        >>> modelReader = binmod()
        >>> modelReader.read("star.bin")
        """
        self._logger.info(f"Evolution started with dt: {dt}, maxTime: {maxTime}")
        with tqdm(total=maxTime, disable=not pbar, desc="Evolution") as pbar:
            while self._t < maxTime:
                dt = min(dt, maxTime - self._t)
                try:
                    useddt = self.timestep(dt)
                    self._logger.evolve(
                        f"step: {self._evolutionarySteps:<10} DT(s): {useddt:<10.2e} AGE(s): {self._t:<10.2e} ETotal(erg): {xp.sum(self._energyGrid):<20.2e}"
                    )
                except EOSInverterError as e:
                    self._logger.error(f"EOS Inverter Error ({e}), stopping evolution")
                    self._logger.error(f"Final energy bounds are {xp.min(self._energyGrid):0.4E} to {xp.max(self._energyGrid):0.4E}")
                    if self.fmodelOut:
                        outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
                        self.save(outPath)
                    break
                if self.imodelOut and self._evolutionarySteps % self.imodelOutCadence == 0:
                    outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
                    self.save(outPath)
                if self._evolutionarySteps % cbc == 0:
                    callback(self, *cargs)

                pbar.update(useddt)
        if self.fmodelOut:
            outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
            self.save(outPath)

    def _cool_star(self, dt):
        """
        Calculates radiative energy loss and removes that energy from the star allowing 
        for radiative cooling.

        Parameters
        ----------
        dt : float
            The timestep to use for the cooling.
        """
        surfaceLuminosity = 4 * xp.pi * self._radius**2 * self.CONST['sigma'] * self._temperatureGrid[-1, :, :]
        surfaceEnergyLoss = surfaceLuminosity * dt
        self._energyGrid[-1, :, :] -= surfaceEnergyLoss

    @property
    def evolutionary_steps(self):
        """
        Returns the number of evolutionary steps taken by the star.
        Returns
        -------
            int: The number of evolutionary steps taken by the star.
        """
        x = self._evolutionarySteps
        return x

    @property
    def age(self):
        """
        Returns the age of the star.
        Returns
        -------
            float: The age of the star.
        """
        x = self._t
        return x

    @property
    def enclosed_mass(self):
        """
        Computes the enclosed mass of the star as a function of radius.

        Returns
        -------
            interp1d: A 1D interpolation function for the enclosed mass.
        """
        enclosedMass = list()
        radii = xp.linspace(0, xp.exp(self._1D_structure.lnR.values.max()))
        integralMass = xp.trapz(self._densityf(radii), radii)
        for r in radii:
            rs = xp.linspace(0, r)
            em = xp.trapz(self._densityf(rs) * (self._mass/integralMass), rs)
            enclosedMass.append(em)
        enclosedMass = xp.array(enclosedMass)
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

    @property
    def temperature(self) -> xp.ndarray:
        """
        Returns the temperature grid for the star.

        Returns
        -------
            xp.ndarray: The temperature grid for the star.
        """

        return self._temperatureGrid.copy()

    @property
    def energy(self) -> xp.ndarray:
        """
        Returns the energy grid for the star.
        """
        return self._energyGrid.copy()

    @property
    def pressure(self) -> xp.ndarray:
        """
        Returns the pressure grid for the star.
        Returns
        -------
            xp.ndarray: The pressure grid for the star.
        """
        return self._pressureGrid.copy()

    @property
    def density(self) -> xp.ndarray:
        """
        Returns the density grid for the star.
        Returns
        -------
            xp.ndarray: The density grid for the star.
        """
        return self._densityGrid.copy()
    
    @property
    def mass(self) -> xp.ndarray:
        """
        Returns the mass grid for the star.
        Returns
        -------
            xp.ndarray: The mass grid for the star.
        """
        return self._massGrid.copy()

    @property
    def surface_temperature_profile(self) -> xp.ndarray:
        """
        Computes the surface temperature profile for the star.

        Returns
        -------
            xp.ndarray: The surface temperature profile for the star.
        """
        surfaceTemperature = self._temperatureGrid[-1, :, :]
        return surfaceTemperature

    def inject_surface_energy(self, energy, theta0, phi0, omega):
        """
        Injects energy into the star at a specified coordinate subtending
        some solid angle.

        Parameters
        ----------
        energy : float
            The energy to inject into the star. This must be given in no log cgs units.
        theta0 : float
            The azimuthal coordinate of the injection point.
        phi0 : float
            The altitudinal coordinate of the injection point.
        omega : float
            The solid angle subtended by the injection point.
        """
        dTheta = xp.arccos(1 - (omega / (2 * xp.pi)))
        dPhi = dTheta * xp.sin(theta0)
        lbi, lbj, lbk = self._get_grid_index(self.r[-1], theta0 - dTheta, phi0 - dPhi)
        rbi, rbj, rbk = self._get_grid_index(self.r[-1], theta0 + dTheta, phi0 - dPhi)
        lui, luj, luk = self._get_grid_index(self.r[-1], theta0 - dTheta, phi0 + dPhi)
        i, j, k = lbi, slice(lbj, rbj), slice(lbk, luk)
        self._energyGrid[i, j, k] += energy

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

    def as_dict(self):
        """
        Returns a dictionary representation of the star.
        Returns
        -------
            dict: A dictionary representation of the star.
        """
        return {
            "temperature": self.temperature,
            "energy": self.energy,
            "pressure": self.pressure,
            "density": self.density,
            "mass": self.mass,
            "surface_temperature_profile": self.surface_temperature_profile,
            "r": self.r,
            "theta": self.theta,
            "phi": self.phi
        }

    def as_pandas(self):
        """
        Returns a pandas DataFrame representation of the star.
        Returns
        -------
            pandas.DataFrame: A DataFrame representation of the star.
        """
        return pd.DataFrame(self.as_dict())

    def save(self, filename: str) -> bool:
        """
        Save to a binary file format. Currently the model format is being defined in the joplin notebook I am using to keep track of development.

        Parameters
        ----------
        filename : str
            The filename to save the star to.

        Returns
        -------
            bool: A flag indicating if the save was successful.
        """
        try:
            self._modelOutputController.save(filename, self)
        except ValueError as e:
            self._logger.error(f"Error saving model: {e}")
            return False
        if not os.path.exists(filename):
            self._logger.error(f"Error saving model: {filename} not found")
            return False
        return True

    def __repr__(self):
        return f"VoxelSphere(mass={self._mass}, model={self._model_path}, X={self._X}, Y={self._Y}, Z={self._Z}, alpha={self.alpha})"
