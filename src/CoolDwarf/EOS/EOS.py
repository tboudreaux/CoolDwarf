from CoolDwarf.EOS.ChabrierDebras2021.EOS import CH21EOS

from CoolDwarf.err import EOSFormattError

def get_eos(path: str, format: str):
    formats = {
            "CD21": CH21EOS
            }

    EOSBuilder = formats.get(format, None)
    if not EOSBuilder:
        raise EOSFormatError(f"{format} is not a defined EOS format. Those are {formats.keys()}")

    return EOSBuilder(path)
