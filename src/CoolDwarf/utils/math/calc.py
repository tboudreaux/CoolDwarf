import numpy as np

def partial_derivative_x(var, dx):
    # Shift the array forward and backward along the x-axis
    forward_shift = np.roll(var, -1, axis=0)
    backward_shift = np.roll(var, 1, axis=0)
    
    partial_x = (forward_shift - backward_shift) / (2.0*dx)
    
    partial_x[0, :, :] = 0  # Assuming zero derivative at the boundary
    partial_x[-1, :, :] = 0  # Assuming zero derivative at the boundary
    
    return partial_x
