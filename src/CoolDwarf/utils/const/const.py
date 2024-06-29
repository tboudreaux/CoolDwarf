"""
const.py -- Constants for CoolDwarf

This module contains the physical constants used in CoolDwarf.

Constants include:
- mH: Hydrogen atomic mass in amu (1.00784)
- mHe: Helium atomic mass in amu (4.002602)
- c: Speed of light in cgs units (2.99792458e10)
- a: Radiation constant in cgs units (7.5646e-15)
- G: Gravitational constant in cgs units (6.6743e-8)
- sigma: Stefan-Boltzmann constant in cgs units (5.670374419e-5)
- R: Ideal gas constant in cgs units (8.3145e7)
- kB: Boltzmann constant in cgs units (1.3807e-16)

Example usage
-------------
>>> from CoolDwarf.utils.const.const import CONST
>>> print(CONST['mH'])
"""
CONST = {
    'mH': 1.00784, #amu 
    'mHe': 4.002602, #amu
    'c': 2.99792458e10, #cgs
    'a': 7.5646e-15, #cgs
    'G': 6.6743e-8, #cgs
    "sigma": 5.670374419e-5, # cgs
    "R": 8.3145e7, #cgs
    "kB": 1.3807e-16 #cgs
}
