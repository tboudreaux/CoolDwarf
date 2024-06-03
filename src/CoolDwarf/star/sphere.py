import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from CoolDwarf.utils.math import make_3d_kernels
from CoolDwarf.utils.math import partial_derivative_x
from CoolDwarf.utils.const import CONST as CoolDwarfCONST
from CoolDwarf.utils.format import pretty_print_3d_array

from CoolDwarf.model import get_model


class VoxelSphere:
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
            D=1,
            k=1,
            dt=1e-4,
            tol={"relax": 1e-6},
            modelFormat='mesa',
            alpha=1.901
            ):
        self._model_path = model
        self._mass = mass
        self._t = t0
        self._X, self._Y, self._Z = X, Y, Z
        self._effectiveMolarMass = self._X * self.CONST['mH'] + self._Y * self.CONST['mHe']
        self._opac = opac
        self.alpha = alpha

        self.radialResolution = radialResolution
        self.azimuthalResolition = azimuthalResolition
        self.altitudinalResolition = altitudinalResolition

        self._D = D
        self._k = k
        self.epsilonH = pressureRegularization

        self._1D_structure = get_model(self._model_path, modelFormat)
        self._radius = np.exp(self._1D_structure.lnR.values.max())
        self._densityf = interp1d(np.exp(self._1D_structure.lnR.values), np.exp(self._1D_structure.lnd.values), bounds_error=False, fill_value=np.exp(self._1D_structure.lnd.values.max()))
        self._temperaturef = interp1d(np.exp(self._1D_structure.lnR.values), np.exp(self._1D_structure.lnT.values), bounds_error=False, fill_value=np.exp(self._1D_structure.lnT.values.max()))

        self._eos = EOS
        self._create_voxel_sphere()
        
        self._evolutionarySteps = 0
        self._dt = dt
        self._tolerances = tol

    def recompute(self):
        self._fill_pressure_energy_grid()
        self.apply_sbc()
        
    @property
    def enclosed_mass(self):
        enclosedMass = list()
        g = list()
        radii = np.linspace(0, np.exp(self._1D_structure.lnR.values.max()))
        integralMass = np.trapz(self._densityf(radii), radii)
        for r in radii:
            rs = np.linspace(0, r)
            em = np.trapz(self._densityf(rs) * (self._mass/integralMass), rs)
            enclosedMass.append(em)
        return interp1d(radii, enclosedMass)
        
    @property
    def radius(self):
        return self._radius

    @property
    def resolution(self):
        return self._resolution

    def _create_voxel_sphere(self):
        self.r = np.linspace(0, self.radius, self.radialResolution)
        self.theta = np.linspace(0, 2*np.pi, self.azimuthalResolition)
        self.phi = np.linspace(0, np.pi, self.altitudinalResolition)

        self._dr = self.r[1] - self.r[0]
        self._dtheta = self.theta[1] - self.theta[0]
        self._dphi = self.phi[1] - self.phi[0]
        
        self.R, self.THETA, self.PHI = np.meshgrid(self.r, self.theta, self.phi, indexing='ij')

        self._temperatureGrid = self._temperaturef(self.R)
        self._densityGrid = self._densityf(self.R)
        self._volumneGrid = abs(self.R**2 * np.sin(self.THETA)* self._dr * self._dtheta * self._dphi)
        self._differentialMassGrid = self._volumneGrid * self._densityGrid

        self._massGrid = self.enclosed_mass(self.R)
        self._forward_EOS()

    def _forward_EOS(self):
        logT = np.log10(self._temperatureGrid)
        logRho = np.log10(self._densityGrid)
        self._pressureGrid = 1e10 * self._eos.pressure(logT, logRho)
        self._energyGrid = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(logT, logRho))

    def Cp(self, delta_t = 1e-5):
        u1 = self._energyGrid
        u2 = 1e13 * ((self._differentialMassGrid/1000) * self._eos.energy(np.log10(self._temperatureGrid + delta_t), np.log10(self._densityGrid)))
        
        cp = (u2 - u1) / delta_t
        cp_specific = cp / self._effectiveMolarMass
        cp_specific[cp_specific == 0] = np.inf
        return cp_specific

    @property
    def gradT(self):
        tGradR, tGradTheta, tGradPhi = np.gradient(self._temperatureGrid, self.r, self.theta, self.phi)
        return (tGradR, tGradTheta, tGradPhi)

    @property
    def gradRadEr(self):
        tGradR, tGradTheta, tGradPhi = self.gradT
        c0 = 4*self.CONST['a'] * self._temperatureGrid**3
        delErR = c0*tGradR
        delErTheta = c0*tGradTheta
        delErPhi = c0*tGradPhi
        return (delErR, delErTheta, delErPhi)

    @property
    def radiative_energy_flux(self):
        opacity = self._opac.kappa(self._temperatureGrid, self._densityGrid)
        c0 = -(self.CONST['c']/(3*opacity*self._densityGrid))
        energyGradient = self.gradRadEr
        fluxGradR = c0 * energyGradient[0]
        fluxGradTheta = c0 * energyGradient[1]
        fluxGradPhi = c0 * energyGradient[2]
        return (fluxGradR, fluxGradTheta, fluxGradPhi)

    @property
    def convective_energy_flux(self):
        tGradR, tGradTheta, tGradPhi = self.gradT
        vR, vTheta, vPhi = self.convective_velocity
        ad = self._adiabatic_grad
        cp = self.Cp()
        density = self._densityGrid

        FradR = density * cp * vR  * (tGradR - ad)
        FradTheta = density * cp * vTheta  * (tGradTheta - ad)
        FradPhi = density * cp * vPhi  * (tGradPhi - ad)
        return (FradR, FradTheta, FradPhi)

    @property
    def energy_flux(self):
        convectiveFlux = self.convective_energy_flux
        radiativeFlux = self.radiative_energy_flux
        return (convectiveFlux, radiativeFlux)

    @property
    def flux_divergence(self):
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
    def dEdt(self):
        fluxDivergence = self.flux_divergence
        dEConvdt = -fluxDivergence[0][0] - fluxDivergence[0][1] - fluxDivergence[0][2]
        dERadt = -fluxDivergence[1][0] - fluxDivergence[1][1] - fluxDivergence[1][2]
        DEDT = dEConvt + dERadt
        return DEDT

    def _update_energy(self, dt):
        dE = - self.dEdt * dt
        self._energyGrid += dE

    def timestep(self, dt):
        self._evolutionarySteps += 1
        self._update_energy(dt)
        self._update_temperature()
        self._update_pressure()
        self._update_density()


    @property
    def convective_overturn_timescale(self):
        vR, vTheta, vPhi = self.convective_velocity
        mixingLength = self.mixing_length

        # Deal with singularity at r==0
        vR[vR == 0] = np.nan
        vTheta[vTheta == 0] = np.nan
        vPhi[vPhi == 0] = np.nan
        tauR = mixingLength / vR
        tauTheta = mixingLength / vTheta
        tauPhi = mixingLength / vPhi

        # Correct the output post singularity to reflect limit
        tauR[np.isnan(vR)] = np.inf
        tauTheta[np.isnan(vTheta)] = np.inf
        tauPhi[np.isnan(vPhi)] = np.inf
        return (tauR, tauTheta, tauPhi)

    @property
    def mixing_length(self):
        return self.alpha*self.pressure_scale_height

    @property
    def pressure_scale_height(self):
        g = self.gravitational_acceleration
        H = self._pressureGrid/(self._densityGrid * g + self.epsilonH)
        return H

    @property
    def convective_velocity(self):
        vc = lambda tg: np.sqrt(self.gravitational_acceleration * self.mixing_length * abs(tg/self._temperatureGrid))
        tGradR, tGradTheta, tGradPhi = self.gradT
        vR = vc(tGradR)
        vTheta = vc(tGradTheta)
        vPhi = vc(tGradPhi)

        return vR, vTheta, vPhi

    @property
    def gravitational_acceleration(self):
        rUse = self.R.copy()
        rUse[self._massGrid == 0] = np.inf # deal with the singularity at r=0
        return (self.CONST['G'] * self._massGrid)/(rUse**2)

    @property
    def _adiabatic_grad(self):
        self._delad = (self._pressureGrid) / (self._densityGrid * self.Cp())
        return self._delad

    @property
    def temperature(self):
        return self._temperature

    def timestep(self, dt, ect = 0.01):
        raise NotImplimentedError()

    def surface_temperature_profile(self, Omega=np.pi, theta=0, i=90):
        raise NotImplimentedError()

    def surface_luminosity_profile(self, Omega=np.pi, theta=0, i=90):
        raise NotImplimentedError()

    def surface_gravity_profile(self, Omaga=np.pi, theta=0, i=90):
        raise NotImplimentedError()
