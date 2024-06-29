import numpy as np
import matplotlib.pyplot as plt

from CoolDwarf.star import VoxelSphere, default_tol
from CoolDwarf.utils import setup_logging
from CoolDwarf.EOS import get_eos
from CoolDwarf.opac import KramerOpac
from CoolDwarf.utils.output import binmod
from CoolDwarf.utils.plot import plot_polar_slice

from time import perf_counter

perf = list()
def monitor(sphere):
    global perf
    perf.append((sphere.evolutionary_steps, sphere.age, perf_counter() - start))


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
start = perf_counter()
sphere.evolve(maxTime = 60*60, pbar=False, dt=30, callback=monitor)

surfaceEnergy = sphere.energy[-1].mean()
sphere.inject_surface_energy(10*surfaceEnergy, 0, 0, np.pi/10)

sphere.evolve(maxTime = 86400 * 7, pbar=False, dt=60, callback=monitor, cbc=10)

# perf = np.array(perf)
# import pandas as pd
# df = pd.DataFrame(perf, columns=["step", "age", "time"])
# df.to_csv("perf-perturb.csv", index=False)

