"""
calc.py -- Calculus related functions for CoolDwarf

This module contains functions related to calculus that are used in CoolDwarf.

Functions include:
- partial_derivative_x: Function to calculate the partial derivative along the x-axis
- compute_partial_derivatives: Function to compute the partial derivatives of a scalar field

Dependencies
------------
- numpy

Example usage
-------------
>>> import numpy as np
>>> from CoolDwarf.utils.math.calc import partial_derivative_x, compute_partial_derivatives
>>> var = np.random.rand(10, 10, 10)
>>> dx = 1.0
>>> partial_x = partial_derivative_x(var, dx)
>>> x = np.linspace(0, 1, 10)
>>> y = np.linspace(0, 1, 10)
>>> z = np.linspace(0, 1, 10)
>>> dfdx, dfdy, dfdz = compute_partial_derivatives(var, x, y, z)
"""
import numpy as np
from typing import Tuple

def partial_derivative_x(var: np.ndarray, dx: float) -> np.ndarray:
    """
    Function to calculate the partial derivative along the x-axis

    Parameters
    ----------
    var : np.ndarray
        Array of values to calculate the partial derivative of
    dx : float
        Spacing between the x-axis points
    
    Returns
    -------
    partial_x : np.ndarray
        Array containing the partial derivative along the x-axis
    """
    # Shift the array forward and backward along the x-axis
    forward_shift = np.roll(var, -1, axis=0)
    backward_shift = np.roll(var, 1, axis=0)
    
    partial_x = (forward_shift - backward_shift) / (2.0*dx)
    
    partial_x[0, :, :] = 0  # Assuming zero derivative at the boundary
    partial_x[-1, :, :] = 0  # Assuming zero derivative at the boundary
    
    return partial_x


def compute_partial_derivatives(scalar_field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to compute the partial derivatives of a scalar field

    Parameters
    ----------
    scalar_field : np.ndarray
        3D array representing the scalar field
    x : np.ndarray
        Array of x-axis values
    y : np.ndarray
        Array of y-axis values
    z : np.ndarray
        Array of z-axis values
    
    Returns
    -------
    dfdx : np.ndarray
        Array containing the partial derivative along the x-axis
    dfdy : np.ndarray
        Array containing the partial derivative along the y-axis
    dfdz : np.ndarray
        Array containing the partial derivative along the z-axis
    """
    dx = np.gradient(x, axis=0)
    dy = np.gradient(y, axis=1)
    dz = np.gradient(z, axis=2)

    # Compute the partial derivatives using NumPy's gradient function
    dfdx = np.gradient(scalar_field, axis=0) / dx
    dfdy = np.gradient(scalar_field, axis=1) / dy
    dfdz = np.gradient(scalar_field, axis=2) / dz

    return dfdx, dfdy, dfdz
