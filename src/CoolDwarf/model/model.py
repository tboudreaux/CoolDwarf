"""
model.py -- General model retreival function for CoolDwarf

This module contains the get_model function, which is used to retrieve the appropriate model object based on the format
of the model file.

Dependencies
------------
- pandas
- CoolDwarf.model.mesa
- CoolDwarf.model.dsep

Example usage
-------------
>>> from CoolDwarf.model.model import get_model
>>> model = get_model("path/to/model/file", "mesa")
>>> print(model)
"""
import pandas as pd

from CoolDwarf.model.mesa import parse_mesa_MOD_file
from CoolDwarf.model.dsep import parse_dsep_MOD_file

def get_model(path: str, format: str) -> pd.DataFrame:
    """
    This function is used to retrieve the appropriate model object based on the format of the model file.
    Available formats are:
    - mesa: MESA MOD files
    - dsep: DSEP MOD files

    Parameters
    ----------
    path : str
        Path to the model file
    format : str
        Format of the model file. Available formats include: mesa for MESA MOD files, dsep for DSEP MOD files.

    Returns
    -------
    model : pd.DataFrame
        DataFrame containing the model data

    Raises
    ------
    SSEModelError
        If the format is not recognized
    """
    formats = {
            "mesa": parse_mesa_MOD_file,
            "dsep": parse_dsep_MOD_file
            }
    modParser = formats.get(format, None)
    if not modParser:
        raise SSEModelError(f"Format {format} is not a valid format. Those are {formats.keys()}")

    return modParser(path)

