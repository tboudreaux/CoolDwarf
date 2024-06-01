import numpy as np

def pretty_print_3d_array(array3D, mask, decimals=3):
    # Determine the maximum number width for formatting
    maxNumWidth = max(len(str(abs(np.round(x, decimals=decimals)))) for x in array3D.flatten() if x != 0)
    
    for layerIndex in range(array3D.shape[0]):
        for rowIndex in range(array3D.shape[1]):
            for colIndex in range(array3D.shape[2]):
                element = np.round(array3D[layerIndex, rowIndex, colIndex], decimals=decimals)
                maskElement = mask[layerIndex, rowIndex, colIndex]
                print(f"{format_number(element, maxNumWidth)}" if maskElement else " " * maxNumWidth, end=" ")
            print()
        print()


def format_number(x, max_width):
    if np.isinf(x):
        s = ' '
        return f"inf{s:<{max_width-3}}"
    if x == 0:
        return f"{x:.{max_width - 2}f}" if max_width > 1 else "0"
    precision = max(0, max_width - len(str(int(x))) - 1)
    formatted_number = "{:.{precision}f}".format(x, precision=precision)
    
    if len(formatted_number) > max_width:
        return formatted_number[:max_width]
    return formatted_number
