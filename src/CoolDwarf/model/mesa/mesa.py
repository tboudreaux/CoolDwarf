"""
mesa.py -- MESA MOD file parser

This module contains a function to parse MESA MOD files. The function reads the file and extracts the metadata and data sections.
Due to the format of the MESA MOD files, the metadata section is read line by line, while the data section is read as a fixed-width file.
The data is then stored in a pandas DataFrame. Finally, because MESA uses D instead of E for scientific notation, the function replaces D with E in the data section.

Dependencies
------------
- pandas

Example usage
-------------
>>> from CoolDwarf.model.mesa.mesa import parse_mesa_MOD_file
>>> df = parse_mesa_MOD_file("path/to/mod/file")
>>> print(df)
"""
import re
import pandas as pd
from io import StringIO

def parse_mesa_MOD_file(filepath: str) -> pd.DataFrame:
    """
    This function reads a MESA MOD file and extracts the metadata and data sections. The data is then stored in a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the MESA MOD file

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the data from the MESA MOD file
    """
    with open(filepath, 'r') as f:
        content = f.read()

    metadataSection = '\n'.join(content.split('\n')[:23])
    metadata = re.finditer(r"(([A-Za-z_\/]+)\s+(.+)\n)[^\n]", metadataSection)

    dataSection = '\n'.join(content.split('\n')[23:-9])
    dataSection = dataSection.replace("D", "E")
    df = pd.read_fwf(StringIO(dataSection), colspecs='infer')
    return df
