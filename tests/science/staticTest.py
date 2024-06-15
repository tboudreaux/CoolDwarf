import numpy as np
import matplotlib.pyplot as plt

from CoolDwarf.star import VoxelSphere, default_tol
from CoolDwarf.utils import setup_logging
from CoolDwarf.EOS import get_eos
from CoolDwarf.opac import KramerOpac
from CoolDwarf.utils.output import binmod
from CoolDwarf.utils.plot import plot_polar_slice

def plot_every_time(sphere):
    fig, ax = plot_polar_slice(sphere, sphere.density)
    fig.savefig(f"density-{sphere._evolutionarySteps}.png")
    plt.close(fig)
    fig, ax = plot_polar_slice(sphere, sphere.temperature)
    fig.savefig(f"temperature-{sphere._evolutionarySteps}.png")
    plt.close(fig)
    fig, ax = plot_polar_slice(sphere, sphere.pressure)
    fig.savefig(f"pressure-{sphere._evolutionarySteps}.png")
    plt.close(fig)
    fig, ax = plot_polar_slice(sphere, sphere.energy)
    fig.savefig(f"energy-{sphere._evolutionarySteps}.png")
    plt.close(fig)

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
    altitudinalResolition=20,
    azimuthalResolition=20,
    cfl_factor = 10,
)

# Model Relaxation
sphere.evolve(maxTime = 60*60, pbar=False, dt=30)
