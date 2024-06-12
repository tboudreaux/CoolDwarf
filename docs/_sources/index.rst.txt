.. CoolDwarf documentation master file, created by
   sphinx-quickstart on Sun Jun  9 13:17:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
CoolDwarf Project
=================

Welcome to the CoolDwarf project. This project aims to provide a easy to use and physically robsut
3D cooling model for fully convective stars. 

CoolDwarf is in very early stages of development and is not yet ready for general or scientific use. However,
we welcome any feedback or contributions to the project.

The CoolDwarf project is hosted on GitHub at https://github.com/tboudreaux/CoolDwarf

===========
Depedenices
===========
CoolDwarf requires the following packages to be installed:

- numpy
- scipy
- matplotlib
- tqdm
- torch

Optional packages:

- cupy

If you are using a CUDA enabled GPU, you can install the cupy package to speed up the calculations
significantly. If you do not have a CUDA enabled GPU, you can still use the package, but it will be
significantly slower. CoolDwarf will automatically detect if cupy is installed and use it if it is available.


============
Installation
============
To install the CoolDwarf package, you can use the following command:

.. code-block:: bash

   git clone https://github.com/tboudreaux/CoolDwarf.git
   cd CoolDwarf
   pip install .


=====
Usage
=====
The CoolDwarf package is designed to be easy to use. The primary entry point for
using the package is the CoolDwarf.star.VoxelSphere class. This class is used to create a 5D
model of a star.

The model is contructed of a grid of equal volume elements spread over a sphere. A MESA model
is used to provide the initial temperature and density profiles of the star. An equation of state
from Chabrier and Debras 2021 is used to calculate the pressure and energy of the star. The radiative opacity
of the star is currentl treated monocromatically and with a simplistic Kramer's opacity model (this is a high 
priority area for improvement).

Evolution of the Cooling model is preformed by calculating the radiative and convective energy gradients 
at each grid point and to find the new energy after some small time step. The Equation of state is then inverted
to update the density, temperature, and pressure of the model. 

Timesteps are dynamicall calculated using the Courant-Friedrichs-Lewy (CFL) condition. The CFL condition is used to ensure
that the timestep is small enough to prevent the model from becoming unstable. If the user wishes to use a fixed timestep,
they can set the cfl_factor to infinity. Alternativley the timestep can be fixed by setting it lower than the CFL condition.

Currently, numerical instabilities exist; however, we are working to resolve these issues. A breif example
of how to use the package is shown below (note that neither the EOS tables nor the MESA model are included in the
repository; however, these are either readily available online (in the case of the EOS model) or can be generated
easily using MESA):

.. code-block:: python

   from CoolDwarf.star import VoxelSphere, default_tol
   from CoolDwarf.utils import setup_logging
   from CoolDwarf.EOS import get_eos
   from CoolDwarf.opac import KramerOpac
   from CoolDwarf.utils.output import binmod

   modelWriter = binmod()

   setup_logging(debug=False)

   EOS = get_eos("EOS/TABLEEOS_2021_Trho_Y0292_v1", "CD21")
   opac = KramerOpac(0.7, 0.02)
   sphere = VoxelSphere(
      8e31,
      "BrownDwarfMESA/BD_TEST.mod",
      EOS,
      opac,
      radialResolution=100,
      altitudinalResolition=100,
      azimuthalResolition=100,
      cfl_factor = 0.4,
   )
   sphere.evolve(maxTime = 60*60*24, pbar=False)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CoolDwarf <CoolDwarf.rst>
   CoolDwarf.star <CoolDwarf.star.rst>
   CoolDwarf.EOS <CoolDwarf.EOS.rst>
   CoolDwarf.EOS.invert <CoolDwarf.EOS.invert.rst>
   CoolDwarf.model <CoolDwarf.model.rst>
   CoolDwarf.model.mesa <CoolDwarf.model.mesa.rst>
   CoolDwarf.opac <CoolDwarf.opac.rst>
   CoolDwarf.utils.misc <CoolDwarf.utils.misc.rst> 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`