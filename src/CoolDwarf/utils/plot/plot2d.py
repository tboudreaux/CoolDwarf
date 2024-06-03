import numpy as np
import matplotlib.pyplot as plt

def visualize_scalar_field(scalar_field, slice_axis='z', slice_index=None, **kwargs):
    if slice_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")
    if slice_index is None:
        slice_index = scalar_field.shape[0] // 2  # Default to the middle slice

    if slice_axis == 'x':
        slice_data = scalar_field[slice_index, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif slice_axis == 'y':
        slice_data = scalar_field[:, slice_index, :]
        xlabel, ylabel = 'X', 'Z'
    else:  # 'z'
        slice_data = scalar_field[:, :, slice_index]
        xlabel, ylabel = 'X', 'Y'

    fig, ax = plt.subplots(**kwargs)
    img = ax.imshow(slice_data, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Scalar Field Value')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Slice along {slice_axis}-axis at index {slice_index}')
    return fig, ax
