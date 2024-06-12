import importlib
import logging

def get_array_module():
    logger = logging.getLogger("CoolDwarf.utils.misc.backend.get_array_module")
    try:
        cupy = importlib.import_module('cupy')
        logger.info("Found Cupy, Using CuPy backend")
        return cupy, True
    except ImportError:
        numpy = importlib.import_module('numpy')
        logger.info("CuPy not found, Using NumPy backend")
        return numpy, False

def get_interpolator():
    try:
        from cupyx.scipy.interpolate import RegularGridInterpolator
    except ImportError:
        from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator
