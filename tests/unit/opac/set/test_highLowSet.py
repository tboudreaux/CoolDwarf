from CoolDwarf.opac import Ferg05Opac
from CoolDwarf.opac import OpalOpac
from CoolDwarf.opac.set import OpacitySet

FOpac = Ferg05Opac(0.7, 0.02)
OOpac = OpalOpac(0.7, 0.02)
OpacSet = OpacitySet(FOpac, OOpac)

print(OpacSet.kappa(5e3, 1e4))
print(OpacSet.kappa(10**4.0, 1e-2))
print(OpacSet.kappa(1e6, 1e-2))
