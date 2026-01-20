import numpy as np


PARAM_UNITS = ["f", "m", "n", "p", "u", "k", "G", "M"]


def nparray_to_params_dict(params_numpy, params_list):
    params_dict = {}
    for i, param in enumerate(params_list):
        params_dict[param] = format_float(params_numpy[i])

    return params_dict

def params_dict_to_nparray(params_dict):
    values = []
    for k, v in params_dict.items():
        print(k, v)
        v_fp = convert_value_to_float(v)
        values.append(v_fp)

    params_numpy = np.array(values)

    return params_numpy




def numpy2params(numpy_array, params_list) -> str:
    """convert numpy array to params string

    Args:
        numpy_array (np.array): numpy array
        params_list (list): list of params

    Returns:
        str: params string
    """
    # params_query = ".param "
    params_query = ""
    for j, param in enumerate(params_list):
        params_query += f"{param}={format_float(numpy_array[j])} "
    return params_query


def params2numpy(params_query, params_list):
    """convert params string to numpy array

    Args:
        params_query (str): params string
        params_list (list): list of params

    Returns:
        np.array: numpy array
    """
    params = params_query.split(" ")
    numpy_array = np.zeros(len(params_list))
    for param in params:
        if "=" in param:
            key, value = param.split("=")
            numpy_array[params_list.index(key)] = convert_value_to_float(value)
    return numpy_array


def format_float(value:float, precision=1) -> str:
    """format float value, convert to string with {precision} decimal points, and add unit if necessary

    Args:
        value (float): a float value
        precision (int, optional): precision of the float value. Defaults to 1.

    Returns:
        str: formatted string
    """
    if value == 0:
        return "0"
    if value < 1e-12:
        return f"{value*1e15:.{precision}f}f"
    if value < 1e-9:
        return f"{value*1e12:.{precision}f}p"
    if value < 1e-6:
        return f"{value*1e9:.{precision}f}n"
    if value < 1e-3:
        return f"{value*1e6:.{precision}f}u"
    if value < 1e3:
        return f"{value:.{precision}f}"
    if value < 1e6:
        return f"{value*1e-3:.{precision}f}k"
    if value < 1e9:
        return f"{value*1e-6:.{precision}f}M"
    if value < 1e12:
        return f"{value*1e-9:.{precision}f}G"
    return f"{value:.{precision}f}"


def convert_value_to_float(value_with_unit:str) -> float:
    """ convert the string value with unit to float

    Args:
        value_with_unit (str): a string value with unit

    Returns:
        float: a float value
    """
    # convert the values to float; eg. 1.0u to 1e-6, 1.0k to 1e3, etc.
    # Find the last character of the value;
    unit = value_with_unit[-1]
    # If the last character is a unit, convert the value to a float;
    if unit in PARAM_UNITS:
        # Convert the value to a float;
        value = float(value_with_unit[:-1])
        # Convert the value to the correct unit;
        if unit == "f":
            value *= 1e-15
        elif unit == "p":
            value *= 1e-12
        elif unit == "n":
            value *= 1e-9
        elif unit == "u":
            value *= 1e-6
        elif unit == "k":
            value *= 1e3
        elif unit == "M":
            value *= 1e6
        elif unit == "G":
            value *= 1e9
        # Update the value in the dictionary;
        return value
    else:
        # If the last character is not a unit, convert the value to a float;
        return float(value_with_unit)
