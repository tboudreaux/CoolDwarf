from CoolDwarf.star import VoxelSphere, default_tol, AdiabaticIdealAtmosphere, SAModel
from CoolDwarf.utils import setup_logging
from CoolDwarf.EOS import get_eos
from CoolDwarf.opac import KramerOpac

setup_logging(debug=True, logName="ModelIntergationTest.log")

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
    cfl_factor = 0.4,
)
atmosphere = AdiabaticIdealAtmosphere(
    sphere,
    Omega=1,
    altitudinalResolution=100,
    azimuthalResolution=10
)
model = SAModel(sphere, atmosphere)

print("Testing timesteping...")
dt = model.timestep(1)
print(f"Struct timestep {dt[0]}, atmoTimestep 1 {dt[1]}, atmoTimestep 2 {dt[2]}")

print("Testing evolution...")
model.evolve()
print("Done!")
