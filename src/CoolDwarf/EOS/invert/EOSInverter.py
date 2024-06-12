"""
EOSInverter.py -- Inverter class for EOS tables

This module contains the Inverter class for EOS tables. The class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions to invert the EOS tables.
Because the inversion problem is non-linear, the Inverter class uses the scipy.optimize.minimize function to find the solution.

Further, because EOSs may not be truley invertible, the Inverter class uses a loss function to find the closest solution to the target energy.
over a limited range of temperatures and densities. This is intended to be a range centered around the initial guess for the inversion and
limited in size by some expected maximum deviation from the initial guess.

Dependencies
------------
- cupy
- torch
- CoolDwarf.err

Example usage
-------------
>>> from CoolDwarf.EOS.invert.EOSInverter import Inverter
>>> from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS
>>> eos = CH21EOS("path/to/eos/table")
>>> inverter = Inverter(eos, TRange, RhoRange)
>>> logTInit, logRhoInit = 7.0, -2.0
>>> newTRange = (6.0, 8.0)
>>> newRhoRange = (-3.0, 0.0)
>>> energy = 1e15
>>> newBounds = (newTRange, newRhoRange)
>>> inverter.set_bounds(newBounds)
>>> logT, logRho = inverter.temperature_density(energy, logTInit, logRhoInit)
"""
import torch
import torch.optim as optim

from CoolDwarf.err import EOSInverterError
from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

def cupy_to_torch(cupy_array):
    return torch.as_tensor(cupy_array.get(), device='cuda')

class Inverter:
    """
    Inverter -- Inverter class for EOS tables

    This class is designed to be used with the CoolDwarf Stellar Structure code, and provides the necessary functions
    to invert the EOS tables. The Inverter class uses PyTorch optimizers to find the solution to the non-linear inversion problem.
    Because EOSs may not be truly invertible, the Inverter class uses a loss function to find the closest solution to the target energy
    over a limited range of temperatures and densities. This is intended to be a range centered around the initial guess for the inversion
    and limited in size by some expected maximum deviation from the initial guess.

    Parameters
    ----------
    EOS : EOS
        EOS object to invert
    TRange : tuple
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
    RhoRange : tuple
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table

    Attributes
    ----------
    EOS : EOS
        EOS object to invert
    _TRange : tuple
        Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
    _RhoRange : tuple  
        Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table
    _bounds : tuple
        Tuple containing the TRange and RhoRange
    
    Methods
    -------
    temperature_density(energy, logTInit, logRhoInit)
        Inverts the EOS to find the temperature and density that gives the target energy
    set_bounds(newBounds)
        Sets the bounds for the inversion
    """
    def __init__(self, EOS):
        """
        Initialize the Inverter class

        Parameters
        ----------
        EOS : EOS
            EOS object to invert
        TRange : tuple
            Tuple containing the minimum and maximum temperature (log10(T)) in the EOS table
        RhoRange : tuple
            Tuple containing the minimum and maximum density (log10(ρ)) in the EOS table
        """
        self.EOS = EOS
        self._bounds = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def temperature_density(self, energy: torch.Tensor, logTInit: torch.Tensor, logRhoInit: torch.Tensor, lr: float = 0.01, num_epochs: int = 1000) -> torch.Tensor:
        if self._bounds != None:
            logTInit = logTInit.to(self.device).requires_grad_(True)
            logRhoInit = logRhoInit.to(self.device).requires_grad_(True)
            energy = energy.to(self.device).requires_grad_(True)

            optimizer = optim.Adam([logTInit, logRhoInit], lr=lr)

            logTInitFlat = logTInit.flatten()
            logRhoInitFlat = logRhoInit.flatten()
            energyFlat = energy.flatten()

            # Reshape bounds to match the flattened grid shape
            T_min_bounds = self._bounds[0, 0].flatten().to(self.device)
            T_max_bounds = self._bounds[0, 1].flatten().to(self.device)
            Rho_min_bounds = self._bounds[1, 0].flatten().to(self.device)
            Rho_max_bounds = self._bounds[1, 1].flatten().to(self.device)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                loss = self._loss(logTInitFlat, logRhoInitFlat, energyFlat)
                loss.backward()
                optimizer.step()

                # Apply bounds for each grid point
                with torch.no_grad():
                    logTInitFlat.data = torch.max(logTInitFlat, T_min_bounds)
                    logTInitFlat.data = torch.min(logTInitFlat, T_max_bounds)
                    logRhoInitFlat.data = torch.max(logRhoInitFlat, Rho_min_bounds)
                    logRhoInitFlat.data = torch.min(logRhoInitFlat, Rho_max_bounds)

                    if loss.item() < 1e-4:
                        break

            if loss.item() >= 1e-4:
                raise EOSInverterError(f"No Inversion found for energy within the given bounds")

            # Reshape the result back to the original grid shape
            logT_result = logTInitFlat.view(logTInit.shape)
            logRho_result = logRhoInitFlat.view(logRhoInit.shape)
            output = torch.stack([logT_result, logRho_result], dim=-1).detach().cpu()

            return output.numpy()
        else:
            raise RuntimeError("Bounds not set for the EOS Inverter")

    def _loss(self, logT: torch.Tensor, logRho: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simple loss function to minimize the difference between the target energy and the EOS energy

        Parameters
        ----------
        logT : torch.Tensor
            Tensor containing the temperature (log10(T)) to evaluate the loss function at
        logRho : torch.Tensor
            Tensor containing the density (log10(ρ)) to evaluate the loss function at
        target : torch.Tensor
            Target energy to invert the EOS to
        
        Returns
        -------
        torch.Tensor
            Loss function value
        """
        energy = self.EOS.energy_torch(logT, logRho)
        loss = torch.abs(energy - target).mean()
        return loss

    def set_bounds(self, tRange, rRange):
        bounds = xp.array([[tRange[0], tRange[1]], [rRange[0], rRange[1]]])
        if CUPY:
            self._bounds = torch.tensor(bounds.get(), device=self.device)
        else:
            self._bounds = torch.tensor(bounds, device=self.device)
