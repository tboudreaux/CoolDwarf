## 1.1.0 (2024-07-25)

### Feat

- **star/atmo**: added major features to atmo, including radiative attenuation
- **utils/interp**: added bilinear and linear interpolation methods
- **opac/set**: added a class to ramp between ferg05 and opal opacities when moving from high to low opacities
- **ferg05**: added ferguson 2005 low temperature opacities
- **opal**: added opal high temp opacity tables and interpolation routines for them
- **atmo.py**: added temperature update code to the atmospher based on the advective, diffusive, and radiative losses over some timestep dt
- **src**: started adding atmospheric model

### Refactor

- **CoolDwarf**: general changes including adding new constants and docs updates
- **opac**: cleaned up unused files

## 1.0.0 (2024-06-15)

### BREAKING CHANGE

- Methods from the EOS inversion method have been dropped and so if they are called anymore the code will fail

### Feat

- **sphere.py**: added evolution callback function
- **EOSInverter.py**: added iterative energy error improvment to arbitary tolerance
- **EOSInverter.py**: New inversion method

### Fix

- **sphere.py**: updated docstrings with more examples
- **plot2d.py**: fixed a bug with backend selection
- **sphere.py**: temp and density now being saved after EOS inversion

### Refactor

- **plot**: brought plot module in line with multiple backend support

## 0.9.0 (2024-06-12)

### Feat

- **logging.py**: Added term output and seperate basic evolutionary output
- **star.py,EOS.py,backend.py,EOSInverter.py**: Multiple Backends (numpy and cupy)
- **binmod.py**: loader added to binmod
- **binmod.py**: binary model output format
- **plot2d.py**: added spherical slice visualization
- **logging.py**: added seperate evolution log
- **err**: added new errors

### Fix

- **__init__.py**: added __init__ to output module
- **EOS.py**: Brought EOS interface into the dual backend system

### Refactor

- **__init__.py**: brought default tolerance dict to the top level of the star module
- **sphere.py**: removed unused imports

### Perf

- **EOS.py,-EOSInverter.py,-sphere.py**: changed from numpy + scipy to cupy + torch for preformanc

## 0.8.0 (2024-06-11)

### Feat

- **sphere.py**: reverse EOS implimented

### Fix

- **star/sphere.py**: added docstrings and fixed phi axis

### Refactor

- **plot2d.py**: plot2d uses spherical coordinates now

## 0.7.0 (2024-06-04)

### Feat

- **EOSInverter**: Added validation to EOS inverter and added more EOS errors
- **EOSInverter.py**: EOS inverter which can optimize temp and density for a given energy in a specific T and rho range
- **sphere.py,EOS.py**: EOS foward operation changed to matrix operation

## 0.6.0 (2024-06-03)

### Feat

- **sphere.py**: Converted from cartesian to sphereical coordinate system

### Fix

- **src/CoolDwarf/utils,src/CoolDwarf/star**: corrected typos in imports
- added basic tests directory

### Refactor

- added gitignore and removed ignorable build files
