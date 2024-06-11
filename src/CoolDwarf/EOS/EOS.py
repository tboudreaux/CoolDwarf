"""
EOS.py -- General EOS retreival function for CoolDwarf

This module contains the get_eos function, which is used to retrieve the appropriate EOS object based on the format
of the EOS table.

Dependencies
------------
- CoolDwarf.EOS.ChabrierDebras2021.EOS
- CoolDwarf.err

Example usage
-------------
>>> from CoolDwarf.EOS.EOS import get_eos
>>> eos = get_eos("path/to/eos/table", "CD21")
"""
from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS

from CoolDwarf.err import EOSFormatError

def get_eos(path: str, format: str):
    """
    This function is used to retrieve the appropriate EOS object based on the format of the EOS table.
    Available formats are:
    - CD21: Chabrier Debras 2021 EOS tables

    Parameters
    ----------
    path : str
        Path to the EOS table
    format : str
        Format of the EOS table. Available formats include: CD21 for Chabrier Debras 2021 EOS tables.
    
    Returns
    -------
    EOS : EOS
        EOS object for the given EOS table

    Raises
    ------
    EOSFormatError
        If the format is not recognized
    """

    formats = {
            "CD21": CH21EOS
            }

    EOSBuilder = formats.get(format, None)
    if not EOSBuilder:
        raise EOSFormatError(format,formats.keys())

    return EOSBuilder(path)
