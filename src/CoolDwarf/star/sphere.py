import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from CoolDwarf.utils.math import make_3d_kernels
from CoolDwarf.utils.const import CONST as CoolDwarfCONST

from CoolDwarf.model import get_model

from CoolDwarf.utils.math import partial_derivative_x
from CoolDwarf.utils.format import pretty_print_3d_array

class VoxelSphere:
    CONST = CoolDwarfConst
    CACHE = {}
    _3DDiffKernels = make_3d_kernels()

    def __init__(self, mass, resolution, model, EOS, opac, t0=0, X=0.75, Y=0.25, Z=0, D=1, k=1, dt=1e-4, tol={"relax": 1e-6}, modelFormat='mesa'):
        self._model_path = model
        self._mass = mass
        self._resolution = resolution
        self._t = t0
        self._X, self._Y, self._Z = X, Y, Z
        self._effectiveMolarMass = self._X * self.CONST['mH'] + self._Y * self.CONST['mHe']
        self._opac = opac

        self._D = D
        self._k = k

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

    def make_dim_arr(self):
        new_x = np.linspace(-self.radius, self.radius, int(self.resolution))
        return new_x

    def apply_sbc(self):
        # Surface boundary condition
        # in the future this should be replaced by an atmospheric boundary condition
        self._temperatureGrid[np.isnan(self.r)] = 0
        self._densityGrid[np.isnan(self.r)] = 0
        self._energyGrid[np.isnan(self.r)] = 0
        self._pressureGrid[np.isnan(self.r)] = 0

    def _create_voxel_sphere(self):
        self.x = self.make_dim_arr()
        self.y = self.make_dim_arr()
        self.z = self.make_dim_arr()
        self.xGrid, self.yGrid, self.zGrid = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.r = np.sqrt(self.xGrid**2 + self.yGrid**2 + self.zGrid**2)
        self.r[self.r > self.radius] = np.nan
        self._dr = np.linalg.norm(np.array([self.xGrid[0,0,0] - self.xGrid[0,1,0], self.yGrid[0,0,0] - self.yGrid[1,0,0], self.zGrid[0,0,0] - self.zGrid[0,0,1]]))
        self._voxelVolume = np.abs((self.x[1] - self.x[0]) * (self.y[1] - self.y[0]) * (self.z[1] - self.z[0]))
        self._voxelFaceArea = np.abs((self.x[1] - self.x[0]) * (self.y[1] - self.y[0]))
        self._voxelSphere = self.r <= self.radius
        self._temperatureGrid = self._temperaturef(self.r)
        self._densityGrid = self._densityf(self.r)
        
        self._massGrid = self.enclosed_mass(self.r)

        self._pressureGrid = np.zeros_like(self.r)
        self._energyGrid = np.zeros_like(self.r)
        self._fill_pressure_energy_grid()

        self.apply_sbc()

    def _fill_pressure_energy_grid(self):
        gridShape = self.r.shape
        for i in range(gridShape[0]):
            for j in range(gridShape[1]):
                for k in range(gridShape[2]):
                    if not self._voxelSphere[i, j, k]:
                        continue # if the region is outside of the sphere do not compute pressure
                    log_temperature = np.log10(self._temperatureGrid[i, j, k])
                    log_density = np.log10(self._densityGrid[i, j, k])
                    self._pressureGrid[i, j, k] = self._eos(log_temperature, log_density, target="pressure")
                    self._energyGrid[i, j, k] = self._eos(log_temperature, log_density, target="U")

    def Cp(self, delta_t = 1e-5):
        u1 = self._energyGrid
        
        # Small temperature increment for numerical derivative
        gridShape = self.r.shape
        u2 = np.zeros_like(u1)
        for i in range(gridShape[0]):
            for j in range(gridShape[1]):
                for k in range(gridShape[2]):
                    if not self._voxelSphere[i, j, k]:
                        continue # if the region is outside of the sphere do not compute pressure
                    log_temperature = np.log10(self._temperatureGrid[i, j, k])
                    log_density = np.log10(self._densityGrid[i, j, k])
                    u2[i, j, k] = self._eos(log_temperature + delta_t, log_density, target="U")
        
        cp = (u2 - u1) / delta_t
        cp_specific = cp / self._effectiveMolarMass
        cp_specific[cp_specific == 0] = np.inf
        return cp_specific

    @property
    def gradT(self):
        tGradX, tGradY, tGradZ = np.gradient(self._temperatureGrid, self._dr, self._dr, self._dr)
        return (tGradX, tGradY, tGradZ)

    @property
    def gradEr(self):
        tGradX, tGradY, tGradZ = self.gradT
        c0 = 4*self.CONST['a'] * self._temperatureGrid**3
        delErX = c0*tGradX
        delErY = c0*tGradY
        delErZ = c0*tGradZ
        return (delErX, delErY, delErZ)

    def radiative_energy_flux(self):
        opacity = self._opac.kappa(self._temperatureGrid, self._densityGrid)
        c0 = -(self.CONST['c']/(3*opacity*self._densityGrid))
        energyGradient = self.gradEr
        fluxGradX = c0 * energyGradient[0]
        fluxGradY = c0 * energyGradient[1]
        fluxGradZ = c0 * energyGradient[2]
        return (fluxGradX, fluxGradY, fluxGradZ)

    def convective_energy_flux(self, alpha=1):
        tGradX, tGradY, tGradZ = self.gradT
        vx, vy, vz = self.convective_velocity(alpha=alpha)
        ad = self._adiabatic_grad
        cp = self.Cp()
        density = self._densityGrid

        FradX = density * cp * vx  * (tGradX - ad)
        FradY = density * cp * vy  * (tGradY - ad)
        FradZ = density * cp * vz  * (tGradZ - ad)
        return (FradX, FradY, FradZ)

    def energy_flux(self, alpha=1):
        convectiveFlux = self.convective_energy_flux(alpha=alpha)
        radiativeFlux = self.radiative_energy_flux()
        return (convectiveFlux, radiativeFlux)

    def flux_divergence(self, alpha=1):
        flux = self.energy_flux(alpha=alpha)
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        delFConvX = partial_derivative_x(flux[0][0], dx)
        delFConvY = partial_derivative_x(flux[0][1], dy)
        delFConvZ = partial_derivative_x(flux[0][2], dz)
        delFRadX = partial_derivative_x(flux[1][0], dx)
        delFRadY = partial_derivative_x(flux[1][1], dy)
        delFRadZ = partial_derivative_x(flux[1][2], dz)

        pretty_print_3d_array(delFConvX)

    def convective_overturn_timescale(self, alpha=1):
        vx, vy, vz = self.convective_velocity(alpha=alpha)
        mixingLength = self.mixing_length(alpha=alpha)
        tauX = mixingLength / vx
        tauY = mixingLength / vy
        tauZ = mixingLength / vz
        return (tauX, tauY, tauZ)

    def mixing_length(self, alpha=1):
        return alpha*self.pressure_scale_height

    @property
    def pressure_scale_height(self):
        return self._pressureGrid/(self._densityGrid * self.gravitational_acceleration)

    def convective_velocity(self, alpha=1):
        vc = lambda tg: np.sqrt(self.gravitational_acceleration * self.mixing_length(alpha=alpha) * (tg/self._temperatureGrid))
        tGradX, tGradY, tGradZ = self.gradT
        vx = vc(tGradX)
        vy = vc(tGradY)
        vz = vc(tGradZ)

        # Convert to spherical and set all but the radial coordinate to 0
        # ******* CURRENTLY THIS IS TURNED OFF (TODO) ********
        x, y, z = self.xGrid, self.yGrid, self.zGrid
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        vr = vx * np.sin(theta) * np.cos(phi) + vy * np.sin(theta) * np.sin(phi) + vz * np.cos(theta)
        vtheta = 0
        vphi = 0

        vx_new = vr * np.sin(theta) * np.cos(phi)
        vy_new = vr * np.sin(theta) * np.sin(phi)
        vz_new = vr * np.cos(theta)
        return (vx, vy, vz)

    @property
    def gravitational_acceleration(self):
        rUse = self.r.copy()
        rUse[self._massGrid == 0] = np.inf # deal with the singularity at r=0
        return (self.CONST['G'] * self._massGrid)/(rUse**2)

    @property
    def _adiabatic_grad(self):
        self._delad = (self._pressureGrid) / (self._densityGrid * self.Cp())
        return self._delad

    def _radiative_grad(self):
        self._delrad = (2 * self._kappa*self._pressure * self._luminosity)/(16 * np.pi * self.CONST['a'] * self.CONST['c'] * self.CONST['G'] * self._mass)
        return self._delrad
        
    @property
    def temperature(self):
        return self._temperature

    def timestep(self, dt, ect = 0.01):
        raise NotImplimentedError()
        pass

    def surface_temperature_profile(self, Omega=np.pi, theta=0, i=90):
        pass

    def surface_luminosity_profile(self, Omega=np.pi, theta=0, i=90):
        pass

    def radial_temperature_profile(self, theta=0, phi=0):
        pass

    def surface_gravity_profile(self, Omaga=np.pi, theta=0, i=90):
        pass
