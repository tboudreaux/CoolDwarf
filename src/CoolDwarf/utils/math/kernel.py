import numpy as np

def make_3d_kernels():
    ck = np.zeros((3, 3, 3))
    ck[1, 1, 1] = 1

    kernels = np.zeros((26, 3, 3, 3))
    counter = 0

    for axis in range(3):  
        for delta in [-1, 1]:  
            k = np.copy(ck)
            k[1 + delta*(axis == 0), 1 + delta*(axis == 1), 1 + delta*(axis == 2)] = -1
            kernels[counter, :, :, :] = k
            counter += 1

    for x in [-1, 1]:
        for y in [-1, 1]:
            k = np.copy(ck)
            k[1 + x, 1 + y, 0] = -1/2
            kernels[counter, :, :, :] = k
            counter += 1
            
            k = np.copy(ck)
            k[1 + x, 0, 1 + y] = -1/2
            kernels[counter, :, :, :] = k
            counter += 1
            
            k = np.copy(ck)
            k[0, 1 + x, 1 + y] = -1/2
            kernels[counter, :, :, :] = k
            counter += 1

    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                k = np.copy(ck)
                k[1 + x, 1 + y, 1 + z] = -1/3
                kernels[counter, :, :, :] = k
                counter += 1

    return kernels
