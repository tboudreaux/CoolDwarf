import logging

from CoolDwarf.utils.misc.backend import get_array_module

xp, CUPY = get_array_module()

def find_closest_values(numList, targetValue):
    if targetValue < min(numList) or targetValue > max(numList):
        raise ValueError("The target value is outside the range of the list.")
    
    sortedList = sorted(numList)
    
    if targetValue in sortedList:
        return targetValue, None
    
    closestLarger = None
    closestSmaller = None
    
    for num in sortedList:
        if num > targetValue:
            closestLarger = num
            break
        closestSmaller = num
    
    return closestLarger, closestSmaller

def linear_interpolate_ndarray(arrays, keys, target):
    fraction = (target - keys[0]) / (keys[1] - keys[0])
    interpolated = arrays[0] + fraction * (arrays[1] - arrays[1])
    return interpolated

def linear_interpolate_dataframes(df_dict, target_key):
    keys = sorted(df_dict.keys())
    
    if target_key in df_dict:
        return df_dict[target_key]
    
    lower_key = max([key for key in keys if key <= target_key], default=None)
    upper_key = min([key for key in keys if key >= target_key], default=None)
    
    if lower_key is None or upper_key is None:
        raise ValueError("Target key is out of the bounds of the dictionary keys.")
    
    df_lower = df_dict[lower_key]
    df_upper = df_dict[upper_key]
    
    if lower_key == upper_key:
        return df_lower
    
    fraction = (target_key - lower_key) / (upper_key - lower_key)
    
    interpolated_df = df_lower + fraction * (df_upper - df_lower)
    
    return interpolated_df

def linear_interpolator(x, x1, x2, Q1, Q2):
    return Q1 + (Q2 - Q1)/(x2 - x1)*(x - x1)

def bilinear_interpolator(x, z, x1, x2, z1, z2, Q00, Q10, Q01, Q11):
    logger = logging.getLogger(__name__)
    A = xp.array([
        [1, x1, z1, x1*z1],
        [1, x1, z2, x1*z2],
        [1, x2, z1, x2*z1],
        [1, x2, z2, x2*z2]
        ])
    Q00 = Q00.values
    Q10 = Q10.values
    Q01 = Q01.values
    Q11 = Q11.values
    result = xp.empty_like(Q00)
    # check if A is singular
    if xp.linalg.det(A) == 0:
        logger.info("Matrix A is singular and cannot be inverted. Checking if composition lies on domain.")
        if x1 == x2 and z1 == z2:
            logger.info("Composition lies on domain")
            return Q00
        elif x1 == x2:
            for i in range(Q00.shape[0]):
                for j in range(Q00.shape[1]):
                    result[i, j] = linear_interpolator(z, z1, z2, Q00[i, j], Q01[i, j])
        elif z1 == z2:
            for i in range(Q00.shape[0]):
                for j in range(Q00.shape[1]):
                    result[i, j] = linear_interpolator(x, x1, x2, Q00[i, j], Q10[i, j])
    else:
        w11 = (x2 - x)*(z2 - z)/(x2 - x1)/(z2 - z1)
        w12 = (x2 - x)*(z - z1)/(x2 - x1)/(z2 - z1)
        w21 = (x - x1)*(z2 - z)/(x2 - x1)/(z2 - z1)
        w22 = (x - x1)*(z - z1)/(x2 - x1)/(z2 - z1)
        for i in range(Q00.shape[0]):
            for j in range(Q00.shape[1]):
                result[i, j] = w11*Q00[i, j] + w12*Q01[i, j] + w21*Q10[i, j] + w22*Q11[i, j]

    return result

def find_closest_indices(x, z, keys):
    x_values = sorted(set(k[0] for k in keys))
    z_values = sorted(set(k[1] for k in keys))
    
    x1 = max([k for k in x_values if k <= x], default=min(x_values))
    x2 = min([k for k in x_values if k >= x], default=max(x_values))
    z1 = max([k for k in z_values if k <= z], default=min(z_values))
    z2 = min([k for k in z_values if k >= z], default=max(z_values))
    
    return x1, x2, z1, z2
