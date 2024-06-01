from CoolDwarf.model.mesa import parse_mesa_MOD_file
from CoolDwarf.model.dsep import parse_dsep_MOD_file

def get_model(path: str, format: str):
    formats = {
            "mesa": parse_mesa_MOD_file,
            "dsep": parse_dsep_MOD_file
            }
    modParser = formats.get(format, None)
    if not modParser:
        raise SSEModelError(f"Format {format} is not a valid format. Those are {formats.keys()}")

    return modParser(path)

