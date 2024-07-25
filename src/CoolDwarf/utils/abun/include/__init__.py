import os
import importlib.resources as pkg

import CoolDwarf.utils.abun.include as include

fileNameLookup = {
        "GS98" : "GS98.abun",
        "AGS09" : "AGS09.abun",
        "GAS07": "GAS07.abun"
        }

def get_abundance_map_path(name: str):
    assert name in fileNameLookup, f"Abundance map not found, valid names are {fileNameLookup.keys()}"
    with pkg.path(include, fileNameLookup[name]) as path:
        return path
