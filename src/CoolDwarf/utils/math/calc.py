import numpy as np

def partial_derivative_x(var, dx):
    # Shift the array forward and backward along the x-axis
    forward_shift = np.roll(var, -1, axis=0)
    backward_shift = np.roll(var, 1, axis=0)
    
    partial_x = (forward_shift - backward_shift) / (2.0*dx)
    
    partial_x[0, :, :] = 0  # Assuming zero derivative at the boundary
    partial_x[-1, :, :] = 0  # Assuming zero derivative at the boundary
    
    return partial_x


def compute_partial_derivatives(scalar_field, x, y, z):
    dx = np.gradient(x, axis=0)
    dy = np.gradient(y, axis=1)
    dz = np.gradient(z, axis=2)

    # Compute the partial derivatives using NumPy's gradient function
    dfdx = np.gradient(scalar_field, axis=0) / dx
    dfdy = np.gradient(scalar_field, axis=1) / dy
    dfdz = np.gradient(scalar_field, axis=2) / dz

    return dfdx, dfdy, dfdz
