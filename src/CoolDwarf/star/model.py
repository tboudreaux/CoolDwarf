
"""
This module contains the SAModel class which is used to evolve a star model
over time. The SAModel class is responsible for managing the evolution of the
star model by calling the timestep method of the structure and atmosphere
objects. The SAModel class also contains the evolve method which is used to
evolve the star model over a specified time period using a specified timestep.

Classes
-------
SAModel
    A class used to evolve a star model over time.

Functions
---------
None

Exceptions
----------
TimestepError
    An exception raised when the remaining time is not used in atmosphere evolution.

Notes
-----
None

Example Usage
-------------
>>> from CoolDwarf.star.model import SAModel
>>> from CoolDwarf.star.atmo import AdiabaticIdealAtmosphere
>>> from CoolDwarf.star.sphere import VoxelSphere
>>> structure = VoxelSphere(...)
>>> atmosphere = AdiabaticIdealAtmosphere(...)
>>> model = SAModel(structure, atmosphere)
>>> model.evolve(maxTime=3.154e+7, dt=86400, pbar=True)
"""
import logging
import os
from tqdm import tqdm

from CoolDwarf.err import TimestepError, EOSInverterError
from CoolDwarf.utils.misc.backend import get_array_module
from CoolDwarf.star.atmo import AdiabaticIdealAtmosphere
from CoolDwarf.star.sphere import VoxelSphere

from typing import Tuple

xp, CUPY = get_array_module()

class SAModel:
    def __init__(
            self,
            structure : VoxelSphere,
            atmosphere : AdiabaticIdealAtmosphere,
            initT : float = 0,
            fModelOut : bool = True,
            iModelOut : bool = False,
            imodelOutCadence : int = 1,
            outputDir : str = "."
            ):
        """
        Initializes the SAModel object.

        Parameters
        ----------
        structure : VoxelSphere
            The structure of the star represented as a VoxelSphere object.
        atmosphere : AdiabaticIdealAtmosphere
            The atmosphere of the star represented as an AdiabaticIdealAtmosphere object.
        initT : float, optional
            The initial time of the star. Default is 0.
        fModelOut : bool, optional
            Flag indicating whether to output the model at each timestep. Default is True.
        iModelOut : bool, optional
            Flag indicating whether to output intermediate models during the evolution. Default is False.
        imodelOutCadence : int, optional
            The cadence at which to output intermediate models. Default is 1.
        outputDir : str, optional
            The directory to save the output models. Default is ".".
        """
        self.structure = structure
        self.atmosphere = atmosphere
        self._t = initT
        self._logger = logging.getLogger("CoolDwarf.star.model.SAModel")
        self._logger.info(f"SAModel initialized with initial time {initT}")
        self.fModelOut = fModelOut
        self.iModelOut = iModelOut
        self.imodelOutCadence = imodelOutCadence
        self._evolutionarySteps = 0
        self._outputDir = outputDir
        self._callbackCalls = 0

    def timestep(
            self,
            dt : float
            ) -> Tuple[float, float, float]:
        """
        Performs a single timestep of the star evolution. The model will first
        time step the structure code and then the atmosphere code. If the
        timestep of the atmosphere code is less than the time step which the
        structure code ended up falling back on, the remaining time will be used
        in a second, "resolver" time step of the atmosphere code. This is used to 
        make sure that the atmosphere code is always using temporally synchronized
        to the structure code.

        Parameters
        ----------
        dt : float
            The target timestep to use for the evolution. Depending on what
            cfl_paremter the structure ends up with the actual used time
            step may be lower than this but the time step will never be higher.

        Returns
        -------
        Tuple[float, float, float]
            A tuple containing the timestep used for the structure, atmosphere
            (first timestep), and atmosphere (second timestep).

        Raises
        ------
        TimestepError
            If the remaining time is not used in atmosphere evolution.
        """
        usedtAtmo2 = 0
        useddtStruct = self.structure.timestep(dt)
        self._logger.info(f"Structure timesteped with {useddtStruct} [s] timestep")
        usedtAtmo = self.atmosphere.timestep(useddtStruct)
        self._logger.info(f"Atmosphere timesteped with {usedtAtmo} [s] timestep")
        if usedtAtmo < useddtStruct:
            remaining = useddtStruct - usedtAtmo
            usedtAtmo2 = self.atmosphere.timestep(remaining)
            if usedtAtmo2 < remaining:
                self._logger.error(f"Remaining time {remaining - usedtAtmo2} [s] is not used in atmosphere evolution.")
                raise TimestepError(f"Remaining time {remaining - usedtAtmo2} [s] is not used in atmosphere evolution.")
        self._evolutionarySteps += 1

        return useddtStruct, usedtAtmo, usedtAtmo2

    def evolve(
            self,
            maxTime : float = 3.154e+7,
            dt : float = 86400,
            pbar : bool = False,
            callback=lambda s: None,
            cbc=1,
            cargs: tuple = ()
            ):
        """
        Evolves the star over a specified time period using a specified timestep.

        Parameters
        ----------
        maxTime : float, optional
            The maximum time to evolve the star. Default is 3.154e+7.
        dt : float, optional
            The timestep to use for the evolution. Default is 86400.
        pbar : bool, optional
            Display a progress bar for the evolution. Default is False.
        callback : function, optional
            A callback function to call at each timestep. Default is a function
            that does nothing. Function will be called after when timestep %
            cbc == 0. The SAModel object will be passed as an argument. The
            function signature must be callback(model, *args).
        cbc : int, optional
            The cadence at which to call the callback function. 1 meaning every time step. 2 would
            be every other timestep and so on...
        cargs : tuple, optional
            Additional arguments to pass to the callback function.
        """
        self._logger.info(f"Evolution started with dt: {dt}, maxTime: {maxTime}")
        with tqdm(total=maxTime, disable=not pbar, desc="Evolution") as pbar:
            while self._t < maxTime:
                dt = min(dt, maxTime - self._t)
                try:
                    a, b, c = self.timestep(dt)
                    if a != b+c:
                        raise TimestepError(f"Remaining atmospheric resolution time {a - b - c} [s] is not used in atmosphere evolution.")
                    self._t += a
                    self._logger.evolve(
                            f"step[struct|atmo]: {self.structure._evolutionarySteps:<6} | {self.atmosphere._evolutionarySteps:<6}, DT(s)[struct|atmo1|atmo2]: {a:<10.2e}|{b:<10.2e}|{c:<10.2e}, AGE(s): {self._t:<10.2e} ETotal(erg): {xp.sum(self.structure._energyGrid) + xp.sum(self.atmosphere._energyGrid):<20.2e}"
                    )
                except EOSInverterError as e:
                    self._logger.error(f"EOS Inverter Error ({e}), stopping evolution")
                    self._logger.error(f"Final energy bounds are {xp.min(self.structure._energyGrid):0.4E} to {xp.max(self.structure._energyGrid):0.4E}")
                    if self.fModelOut:
                        outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
                        self.save(outPath)
                    break
                
                if self.iModelOut and self._evolutionarySteps % self.imodelOutCadence == 0:
                    outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
                    self.save(outPath)
                if self._evolutionarySteps % cbc == 0:
                    callback(self, *cargs)
                    self._callbackCalls += 1 

                pbar.update(a)
        if self.fModelOut:
            outPath = os.path.join(self._outputDir, f"star_{self._t}.bin")
            self.save(outPath)

    def save(self, path):
        """
        Saves the current state of the star model to a file.

        Parameters
        ----------
        path : str
            The path to save the model file.
        """
        ...
