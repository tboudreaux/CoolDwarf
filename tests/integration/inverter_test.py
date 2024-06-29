from CoolDwarf.EOS import get_eos
from CoolDwarf.EOS.invert import Inverter
from CoolDwarf.opac import KramerOpac
from CoolDwarf.star import VoxelSphere

from time import perf_counter



eos = get_eos("../science/TABLEEOS_2021_Trho_Y0292_v1", "CD21")
iEOS = Inverter(eos)
opac = KramerOpac(0.7, 0.02)
sphere = VoxelSphere(
    8e31,
    "../science/BD_TEST.mod",
    eos,
    opac,
    radialResolution=100,
    altitudinalResolition=100,
    azimuthalResolition=100,
    cfl_factor = 0.4,
)
# sphere._update_energy(86400)
i, j, k = slice(100), slice(100), slice(100)

U = (1000 * sphere.energy[i, j, k])/(1e13 * sphere._differentialMassGrid[i, j, k])
call = lambda : iEOS.temperature_density(U, sphere.temperature[i, j, k], sphere.density[i, j, k])
start = perf_counter()
try:
    r =call()
except ValueError:
    print("Failed")
end = perf_counter()
print("Time taken: ", end-start)

