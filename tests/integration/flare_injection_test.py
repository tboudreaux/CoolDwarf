from CoolDwarf.star import VoxelSphere, default_tol, AdiabaticIdealAtmosphere, SAModel
from CoolDwarf.utils import setup_logging
from CoolDwarf.EOS import get_eos
from CoolDwarf.opac import KramerOpac
from CoolDwarf.utils.misc import color_string
from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

def injectFlare(model):
    if model._evolutionarySteps == 10:
        print(color_string("Injecting flare...", "yellow"))
        model.atmosphere._energyGrid[:, 0:2, 0:2] += 1e32
    print(color_string(f"atmosphere layer 1 temperature std: {xp.mean(model.atmosphere._temperatureGrid[0])}", "green"))

setup_logging(debug=True, logName="FlareInjectionIntergationTest.log")

EOS = get_eos("../science/TABLEEOS_2021_Trho_Y0292_v1", "CD21")
opac = KramerOpac(0.7, 0.02)
tol = default_tol()
tol['volCheck'] = 0.01
sphere = VoxelSphere(
    8e31,
    "../science/BD_TEST.mod",
    EOS,
    opac,
    radialResolution=10,
    altitudinalResolition=10,
    azimuthalResolition=10,
    tol=tol,
    cfl_factor = 10,
)
atmosphere = AdiabaticIdealAtmosphere(
    sphere,
    Omega=1,
    altitudinalResolution=1000,
    azimuthalResolution=1000
)
atmosphere.inject_energy(xp.pi/2, 0, 500 * 1e5, 1e17)
exit()
model = SAModel(sphere, atmosphere)

print("Testing timesteping...")
dt = model.timestep(1)
print(f"Struct timestep {dt[0]}, atmoTimestep 1 {dt[1]}, atmoTimestep 2 {dt[2]}")

print("Testing evolution...")
model.evolve(callback = injectFlare, cbc=1)
print("Done!")
