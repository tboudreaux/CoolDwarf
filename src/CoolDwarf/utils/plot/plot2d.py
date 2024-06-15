import numpy as np
import matplotlib.pyplot as plt

from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

def plot_polar_slice(sphere, data, phi_slice=0, theta_offset=0):
    if CUPY:
        phi_idx = (xp.abs(sphere.PHI[0, 0, :].get() - phi_slice)).argmin()
        r_slice = sphere.R[:, :, phi_idx].get()
        theta_slice = sphere.THETA[:, :, phi_idx].get() + theta_offset
        data_slice = data[:, :, phi_idx].get()
    else:
        phi_idx = (xp.abs(sphere.PHI[0, 0, :] - phi_slice)).argmin()
        r_slice = sphere.R[:, :, phi_idx]
        theta_slice = sphere.THETA[:, :, phi_idx] + theta_offset
        data_slice = data[:, :, phi_idx]
    
    
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': 'polar'})
    c = ax.pcolormesh(theta_slice, r_slice, data_slice, shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    return fig, ax
