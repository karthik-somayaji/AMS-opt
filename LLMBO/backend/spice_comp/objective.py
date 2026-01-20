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
    ugf = f_vals['ugf']
    gain = f_vals['gain']
    hyst_err = f_vals['hyst_err']
    offset = f_vals['offset']

    # get the normalization values
    ugf_range = ckt['ugf_norm']
    gain_range = ckt['gain_norm']
    hyst_err_range  = ckt['hyst_err_norm']
    offset_range  = ckt['offset_norm']
    
    # normalize the objective function values
    if(f_vals['ugf']) <= 0:
        f_vals['ugf'] = ugf_range[0]
        ugf = ugf_range[0]
    f_vals['ugf_normed'] = (np.log(ugf) - np.log(ugf_range[0])) / (np.log(ugf_range[1]) - np.log(ugf_range[0]))
    #print('ugf', f_vals['ugf'])
    #print('ugf-normed',f_vals['ugf_normed'])
    f_vals['gain_normed'] = (gain - gain_range[0]) / (gain_range[1] - gain_range[0])
    f_vals['hyst_err_normed'] = (np.abs(hyst_err) - hyst_err_range[0]) / (hyst_err_range[1] - hyst_err_range[0])
    f_vals['offset_normed'] = (np.abs(offset) - offset_range[0]) / (offset_range[1] - offset_range[0])

    # # clip the normalized values to [0, 1]
    f_vals['ugf_normed'] = max(0, min(1, f_vals['ugf_normed']))
    f_vals['gain_normed'] = max(0, min(1, f_vals['gain_normed']))
    # f_vals['cmrr_normed'] = max(0, min(1, f_vals['cmrr_normed']))
    # f_vals['pm_normed'] = max(0, min(1, f_vals['pm_normed']))

    if np.abs(f_vals['offset_normed']) > 1.0:
        f_vals['offset_normed'] = 0.0
    else:
        f_vals['offset_normed'] = 1 - np.abs(f_vals['offset_normed'])

    if np.abs(f_vals['hyst_err_normed']) > 1.0:
        f_vals['hyst_err_normed'] = 0.0
    else:
        f_vals['hyst_err_normed'] = 1 - np.abs(f_vals['hyst_err_normed'])


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
        # if((merit == 'hyst_err') or (merit == 'offset')):
        #     fom -= ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
        # else:
        #     fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
        fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
    f_vals['fom'] = fom
    return f_vals