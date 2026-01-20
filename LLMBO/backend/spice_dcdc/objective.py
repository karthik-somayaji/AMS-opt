import numpy as np

def read_results(f_val, val_name, args, check_specs=False):
    '''
    read f_val as a float nubmer: return the float number if it is a float number, otherwise return the failed value in args
    Args: f_val (str), string containing the float number
          val_name (str), name of the value
          args (dict), dictionary containing circuit information
            check_specs (bool), whether to check the specs of the circuit
    '''
    try:
        f_val = float(f_val)
        if check_specs:
            if val_name is 'pow':
                if f_val > args[f'{pow}_spec']:
                    return args[f'{val_name}_failed']
            else:
                if f_val < args[f'{val_name}_spec']:
                    return args[f'{val_name}_failed']
        return float(f_val)
    except ValueError:
        print(f'Failed to read {val_name} from the file')
        return args[f'{val_name}_failed']

def normalize_fvals(f_vals: dict, ckt: dict):
    '''
    Normalize the objective function values
    Args: f_vals (dict), dictionary containing objective function values
          ckt (dict), dictionary containing circuit information
    Returns: f_vals (dict), dictionary containing normalized objective function values
    '''
    # get the objective function values
    power_output = f_vals['power_output']
    stability = f_vals['stability']
    output_voltage_difference = f_vals['output_voltage_difference']

    # get the normalization values
    power_output_range = ckt['power_output_norm']
    stability_range = ckt['stability_norm']
    output_voltage_difference_range  = ckt['output_voltage_difference_norm']
    
    # normalize the objective function values
    f_vals['power_output_normed'] = -1*(np.log(power_output) - np.log(power_output_range[0])) / (np.log(power_output_range[1] )- np.log(power_output_range[0]))
    f_vals['stability_normed'] = 1 - np.abs((stability - stability_range[0]) / (stability_range[1] - stability_range[0]))

    if np.abs(output_voltage_difference) >= ckt['output_voltage_difference_failed']:
        f_vals['output_voltage_difference_normed'] = -2
    else:
        f_vals['output_voltage_difference_normed'] = -1*(np.abs(output_voltage_difference) - output_voltage_difference_range[0]) / (output_voltage_difference_range[1] - output_voltage_difference_range[0])

    # # clip the normalized values to [0, 1]
    # f_vals['gbw_normed'] = max(0, min(1, f_vals['gbw_normed']))
    # f_vals['gain_normed'] = max(0, min(1, f_vals['gain_normed']))
    # f_vals['cmrr_normed'] = max(0, min(1, f_vals['cmrr_normed']))
    # f_vals['pm_normed'] = max(0, min(1, f_vals['pm_normed']))

    


    return f_vals

def objective(f_vals: dict, ckt: dict):
    '''
    Calculate the objective function value
    Args: f_vals (dict), dictionary containing normalized objective function values
          ckt (dict), dictionary containing circuit information
    Returns: objective (float), objective function value
    '''
    # normalize the objective function values
    f_vals = normalize_fvals(f_vals, ckt)
    # merits
    #merits = ['ugf', 'gain']
    merits = ckt['metrics_list']
    fom = 0
    for merit in merits:
        if((merit == 'hyst_err') or (merit == 'offset')):
            fom -= ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
        else:
            fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
    f_vals['fom'] = fom
    return f_vals