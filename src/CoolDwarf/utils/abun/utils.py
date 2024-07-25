import CoolDwarf.utils.abun as miscAbun

from CoolDwarf.utils.abun.parse import open_and_parse

import numpy as np

try:
    import importlib.resources as pkg
except ImportError: #For python < 3.7
    import importlib_resources as pkg

def mfrac_to_a(mfrac,amass,X,Y):
	"""
	Convert mass fracition of a given element to a for that element at a given
	hydrogen mass fraction using the equation

	.. math::

		a(i) = \\log(1.008) + \\log(F_{i}) - \\left[\\log(X) + \\log(m_{i}) \\right]

	Where :math:`F_{i}` is the mass fraction for the :math:`i^{th}` element and
	:math:`m_{i}` is the mass fraction for the :math:`i^{th}` element.

	Parameters
	----------
		mfrac : float
			Mass fraction of the ith element.
		amass : float
			Mass of the ith element in atomic mass units.
		X : float
			Hydrogen mass fraction
		Y : float
			Helium mass fraction, will be used as reference if X = 0

	Returns
	-------
		a : float
			a for the ith element
	"""
	if mfrac == 0:
		a = -99.99
	else:
		if X != 0:
			a = 12 + np.log(mfrac/(amass*(X/1.008)))/(np.log(2)+np.log(5))
		else:
			if Y == 0:
				raise ValueError("Error! X and Y cannot both be equal to 0!")
			a = 10.93 + np.log(mfrac/(amass*(Y/4.002602)))/(np.log(2)+np.log(5))

	return a

def a_to_mfrac(a,amass,X):
    """
    Convert a for the ith element to the mass fraction of the ith element using
    the equation:

    .. math::

        F_{i} = \\left[\\frac{Xm_{i}}{1.008}\\right]\\times 10^{a(i)-12}

    Where :math:`F_{i}` is the mass fraction for the :math:`i^{th}` element and
    :math:`m_{i}` is the mass fraction for the :math:`i^{th}` element.

    Parameters
    ----------
        a : float
            a for the ith element
        amass : float
            Mass of the ith element in atomic mass units.
        X : float
            Hydrogen mass fraction

    Returns
    -------
        mfrac : float
            mass fraction for the ith element
    """
    mfrac = ((X*amass)/1.008)*10**(a-12)
    return mfrac

def get_atomic_masses():
    """
    Return a dict of atomic masses from Hydrogen all the way to plutonium

    Returns
    -------
        amasses : dict of floats
            Dicionary of atomic masses in atomic mass units indexed by elemental
            symbol.
    """
    amasses = {
        'H': 1.008,
        'He': 4.003,
        'Li': 6.941,
        'Be': 9.012,
        'B': 10.81,
        'C': 12.01,
        'N': 14.01,
        'O': 16.00,
        'F': 19.00,
        'Ne': 20.18,
        'Na': 22.99,
        'Mg': 24.31,
        'Al': 26.98,
        'Si': 28.09,
        'P': 30.97,
        'S': 32.07,
        'Cl': 35.45,
        'Ar': 39.95,
        'K': 39.10,
        'Ca': 40.08,
        'Sc': 44.96,
        'Ti': 47.87,
        'V': 50.94,
        'Cr': 52.00,
        'Mn': 54.94,
        'Fe': 55.85,
        'Co': 58.93,
        'Ni': 58.69,
        'Cu': 63.55,
        'Zn': 65.38,
        'Ga': 69.72,
        'Ge': 72.63,
        'As': 74.92,
        'Se': 78.97,
        'Br': 79.90,
        'Kr': 83.80,
        'Rb': 85.47,
        'Sr': 87.62,
        'Y': 88.91,
        'Zr': 91.22,
        'Nb': 92.91,
        'Mo': 95.95,
        'Tc': 98.00,
        'Ru': 101.1,
        'Rh': 102.9,
        'Pd': 106.4,
        'Ag': 107.9,
        'Cd': 112.4,
        'In': 1148,
        'Sn': 118.7,
        'Sb': 121.8,
        'Te': 127.6,
        'I': 126.9,
        'Xe': 131.3,
        'Cs': 132.9,
        'Ba': 137.3,
        'La': 138.6,
        'Ce': 140.1,
        'Pr': 149.9,
        'Nd': 144.2,
        'Pm': 145.0,
        'Sm': 150.4,
        'Eu': 152.0,
        'Gd': 157.3,
        'Tb': 158.9,
        'Dy': 162.5,
        'Ho': 164.9,
        'Er': 167.3,
        'Tm': 168.9,
        'Yb': 173.04,
        'Lu': 175.0,
        'Hf': 178.5,
        'Ta': 180.9,
        'W': 183.8,
        'Re': 186.2,
        'Os': 190.2,
        'Ir': 192.2,
        'Pt': 195.1,
        'Au': 197.0,
        'Hg': 200.6,
        'Tl': 204.4,
        'Pb': 207.2,
        'Bi': 209.0,
        'Po': 209,
        'At': 210,
        'Rn': 222,
        'Fr': 223,
        'Ra': 226,
        'Ac': 227,
        'Th': 232,
        'Pa': 231,
        'U': 238
    }
    return amasses

def est_feh_from_Z_and_X(abunTable : dict, Xt : float, Zt : float) -> float:
    """
    Analytically estimate feh from Z and X

    Parameters
    ----------
        abunTable : dict
            Abundance Table dictionary in the form described in the docs for
            pysep.misc.abun.util.open_and_parse.
        Xt : float
            Target X to move to
        Zt : float
            Target Z to move to.

    Returns
    -------
        FeH : float
            [Fe/H] value to add to every a(i) for every tracked element i where
            i > 2 (i.e all the metals).
    """
    if Zt != 0:
        Xi = abunTable['AbundanceRatio']['X']
        Zi = abunTable['AbundanceRatio']['Z']
        Yt = 1-(Xt+Zt)
        rAbun = abunTable['RelativeAbundance']
        mfDict = {key:rAbun[key]['m_f'] for key in rAbun}
        aDict = {key:rAbun[key]['a'] for key in rAbun}
        mDict = get_atomic_masses()

        dZ = Zt-Zi
        zfc = dZ/Zi
        # scale the mass fractions by the same fractional change that the total
        # metallicity changed by.
        NewFracs = {key:item*(1+zfc)
                    for key, item in mfDict.items()
                    if key not in ['H', 'He']}
        newADict = {sym:mfrac_to_a(frac, mDict[sym], Xt, Yt)
                    for sym, frac in NewFracs.items()
                    if sym not in ['H', 'He']}
        diffs = [a-aDict[sym]
                for sym, a in newADict.items()]
        FeH = np.median(diffs)
    else:
        FeH = -99.99
    return FeH

def gen_abun_map(abunTable):
    """
    Generate an analytic mapping between X, Y, Z and FeH given an abundance
    table.

    Parameters
    ----------
        abunTable : str
            Path of checmical abundance table to use for composition. Format of
            this table is defined in the ext module documentation.

    Returns
    -------
        MetalAbunMap : function(X,Y,Z) -> (Fe/H,0.0,a(He))
            Function build from interpolation of a grid of FeH, alpha/Fe, and
            a(He) which will returned the set of those values giving the
            composition most similar to an input X, Y, and Z.

    """
    abunDict = open_and_parse(abunTable)
    Yi = abunDict['AbundanceRatio']['Y']
    MetalAbunMap = lambda X, Y, Z: (
            est_feh_from_Z_and_X(abunDict, X, Z),
            0,
            mfrac_to_a(Y, 4.002602, X, Yi))
    return MetalAbunMap

def parse_abundance_map() -> np.ndarray:
    """
    Parse Hydroge, Helium, and metal mass fraction out of a csv where each row
    is one composition, the first column is X, second is Y, and the third is Z.
    Comments may be included in the file if the first non white space charectar
    on the line is a hash.

    Returns
    -------
        pContents : np.ndarray(shape=(n,3))
            numpy array of all the compositions of length n where n is the
            number of rows whos first non white space charectar was not a hash.
            For a dsep n=126. Along the second axis the first column is X, the
            second is Y, and the third is Z.

    """
    with pkg.path(miscAbun, "DSEPAbundanceMap") as path:
        with open(path, 'r') as f:
            contents = f.read().split('\n')
            pContents = np.array([[float(y) for y in x.split(',')]
                                   for x in contents
                                  if x != '' and x.lstrip()[0] != '#'])
    return pContents
