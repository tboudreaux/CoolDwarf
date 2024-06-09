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

    fig, ax = plt.subplots(**kwargs, subplot_kw={'projection': 'polar'})
    img = ax.imshow(slice_data, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Scalar Field Value')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Slice along {slice_axis}-axis at index {slice_index}')
    return fig, ax
