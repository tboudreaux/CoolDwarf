from CoolDwarf.star import VoxelSphere, default_tol
from CoolDwarf.utils import setup_logging
from CoolDwarf.EOS import get_eos
from CoolDwarf.opac import KramerOpac
from CoolDwarf.utils.output import binmod

modelWriter = binmod()

setup_logging(debug=False)

EOS = get_eos("../../../EOS/TABLEEOS_2021_Trho_Y0292_v1", "CD21")
opac = KramerOpac(0.7, 0.02)
sphere = VoxelSphere(
    8e31,
    "../../../BrownDwarfMESA/BD_TEST.mod",
    EOS,
    opac,
    radialResolution=100,
    altitudinalResolition=100,
    azimuthalResolition=100,
    cfl_factor = 0.4,
)
sphere.evolve(maxTime = 60*60*24, pbar=False, dt=1.5)
