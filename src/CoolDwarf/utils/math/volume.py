import numpy as np
from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.err import VolumeError

import logging
from typing import Tuple

xp, CUPY = get_array_module()

def spherical_grid_equal_volume(
        numRadial: int,
        numTheta: int,
        numPhi: int,
        radius: float,
        tol: float,
        minR: float= 0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Generate points within a sphere with equal volume using a stratified sampling approach.
    Returns radius, theta, phi as meshgrids and volume elements for each point.

    Parameters
    ----------
    numRadial : int
        Number of radial segments.
    numTheta : int
        Number of azimuthal segments.
    numPhi : int
        Number of altitudinal segments.
    radius : float
        Radius of the sphere.
    tol : float
        Tolerance for volume error. The maximum fractional error in the volume
        difference between the true volume and the discrete volume.
    minR : float, default 0
        Internal radius of the sphere. If 0 then the sphere is a solid sphere.
        if minR > 0 then the sphere is a shell.
    
    Returns
    -------
    R, THETA, PHI, r, theta, phi, volumeElements, volumeError : tuple
        A tuple of meshgrids for the radial, azimuthal, and altitudinal
        positions. Additionally, the volume elements for each point in the grid
        and the volume error.

    Raises
    ------
    VolumeError
        If the volume error is greater than the tolerance.

    Example Usage
    -------------
    >>> R, THETA, PHI, r, theta, phi, dV, err  = spherical_grid_equal_volume(10, 10, 10, 1)
    """
    logger = logging.getLogger("CoolDwarf.utils.math.volume")
    rEdges = xp.linspace(minR, radius, numRadial + 1)
    r = (rEdges[:-1] + rEdges[1:]) / 2 
    dr = xp.diff(rEdges) 

    thetaEdges = xp.linspace(0, 2 * xp.pi, numTheta + 1)
    theta = (thetaEdges[:-1] + thetaEdges[1:]) / 2 
    dtheta = xp.diff(thetaEdges) 

    phiEdges = xp.linspace(-xp.pi/2, xp.pi/2, numPhi + 1)
    phi = (phiEdges[:-1] + phiEdges[1:]) / 2 
    dphi = xp.diff(phiEdges)  

    R, THETA, PHI = xp.meshgrid(r, theta, phi, indexing='ij')
    dR, dTHETA, dPHI = xp.meshgrid(dr, dtheta, dphi, indexing='ij')

    volumeElements = xp.abs((R ** 2) * xp.sin(PHI) * dR * dTHETA * dPHI)

    discreteVolume = xp.sum(volumeElements)
    outerVolume = 4/3 * xp.pi * radius**3
    innerVolume = 4/3 * xp.pi * minR**3
    trueVolume = outerVolume - innerVolume
    volumeError = abs(discreteVolume - trueVolume) / trueVolume
    
    # Check if the fractional error in the volume difference is within tol['volCheck']
    if volumeError > tol:
        raise VolumeError(f"Volume error is greater than tolerance: {abs(discreteVolume - trueVolume) / trueVolume}")
    logger.info(f"Volume error: {abs(discreteVolume - trueVolume) / trueVolume} is within tolerance ({tol})")

    return R, THETA, PHI, r, theta, phi, volumeElements, volumeError
