from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.utils.const import CONST
from CoolDwarf.typing import Arithmetic

import logging

xp, CUPY = get_array_module()

def plank_iradience(nu, T):
    return (2*CONST.h*nu**3 / (CONST.c**2)) * 1/(xp.exp((CONST.h*nu)/(CONST.kB*T))-1)

def plank_flux_isotropic(nu, T):
    return 4*xp.pi * plank_iradience(nu, T)

def plank_flux_collimated(nu, T, omega=2*xp.pi):
    return omega * plank_iradience(nu, T)

def optical_depth(opacity, density, ds):
    print("Entry: ", opacity[:, 0, 0], density[:, 0, 0], ds[:, 0, 0])
    return xp.sum(opacity * density * ds)

def attenuate_flux(f0, tau):
    return f0 * xp.exp(-tau)

