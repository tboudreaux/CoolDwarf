import re
import pandas as pd
from io import StringIO

def parse_mesa_MOD_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    metadataSection = '\n'.join(content.split('\n')[:23])
    metadata = re.finditer(r"(([A-Za-z_\/]+)\s+(.+)\n)[^\n]", metadataSection)

    dataSection = '\n'.join(content.split('\n')[23:-9])
    dataSection = dataSection.replace("D", "E")
    df = pd.read_fwf(StringIO(dataSection), colspecs='infer')
    return df
