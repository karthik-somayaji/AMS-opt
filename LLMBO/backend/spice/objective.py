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
            if val_name == 'pow':
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
    gbw = f_vals['gbw']
    gain = f_vals['gain']
    cmrr = f_vals['cmrr']
    pm = f_vals['pm']
    pow = f_vals['pow']

    # get the normalization values
    gbw_range = ckt['gbw_norm']
    gain_range = ckt['gain_norm']
    cmrr_range = ckt['cmrr_norm']
    pm_range = ckt['pm_norm']
    pow_range = ckt['pow_norm']

    # normalize the objective function values
    f_vals['gbw_normed'] = (gbw - gbw_range[0]) / (gbw_range[1] - gbw_range[0])
    f_vals['gain_normed'] = (gain - gain_range[0]) / (gain_range[1] - gain_range[0])
    f_vals['cmrr_normed'] = (cmrr - cmrr_range[0]) / (cmrr_range[1] - cmrr_range[0])
    f_vals['pm_normed'] = (pm - pm_range[0]) / (pm_range[1] - pm_range[0])
    f_vals['pow_normed'] = (pow - pow_range[0]) / (pow_range[1] - pow_range[0])

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
    merits = ['gbw', 'gain', 'cmrr', 'pm']
    fom = 0
    for merit in merits:
        fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
    f_vals['fom'] = fom
    return f_vals