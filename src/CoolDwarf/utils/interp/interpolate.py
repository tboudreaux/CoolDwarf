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
