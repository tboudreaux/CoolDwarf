import numpy as np
import matplotlib.pyplot as plt

def visualize_scalar_field(scalar_field, slice_axis='z', slice_index=None, **kwargs):
    if slice_axis not in ['i', 'j', 'k']:
        raise ValueError("slice_axis must be 'i', 'j', or 'k'")
    if slice_index is None:
        slice_index = scalar_field.shape[0] // 2  # Default to the middle slice

    if slice_axis == 'i':
        slice_data = scalar_field[slice_index, :, :]
        xlabel, ylabel = r'$\theta$', r'$\phi$'
    elif slice_axis == 'j':
        slice_data = scalar_field[:, slice_index, :]
        xlabel, ylabel = r'$r$', r'$\phi$'
    else:  # 'k'
        slice_data = scalar_field[:, :, slice_index]
        xlabel, ylabel = r'$r$', r'$\theta$'

    if slice_axis != 'i':
        fig, ax = plt.subplots(**kwargs, subplot_kw={'projection': 'polar'})
    else:
        fig, ax = plt.subplots(**kwargs)
    img = ax.imshow(slice_data, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Scalar Field Value')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Slice along {slice_axis}-axis at index {slice_index}')
    return fig, ax

def plot_polar_slice(sphere, data, phi_slice=0, theta_offset=0, fname="polar_slice.png"):
    phi_idx = (np.abs(sphere.PHI[0, 0, :].get() - phi_slice)).argmin()
   
    r_slice = sphere.R[:, :, phi_idx].get()
    theta_slice = sphere.THETA[:, :, phi_idx].get() + theta_offset
    data_slice = data[:, :, phi_idx].get()
    
    
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': 'polar'})
    c = ax.pcolormesh(theta_slice, r_slice, data_slice, shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.savefig(fname)
    plt.close()
