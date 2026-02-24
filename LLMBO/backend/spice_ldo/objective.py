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
            if val_name == 'pow':
                if f_val > args['pow_spec']:
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
    q_curr = float(f_vals['q_curr'])
    stability = float(f_vals['stability'])
    output_voltage_difference = float(f_vals['output_voltage_difference'])

    # For minimize-metrics, normalize as a clipped "meets spec" score:
    # - score = 1.0 when value <= spec
    # - score < 1.0 when value > spec
    eps = 1e-15
    q_curr_spec = float(ckt.get('q_curr_spec', 0.0))
    ovd_spec = float(ckt.get('output_voltage_difference_spec', 0.0))

    q_curr_den = max(q_curr, eps)
    ovd_den = max(abs(output_voltage_difference), eps)

    if q_curr_spec > 0:
        f_vals['q_curr_normed'] = min(q_curr_spec / q_curr_den, 1.0)
    else:
        f_vals['q_curr_normed'] = 0.0

    if ovd_spec > 0:
        f_vals['output_voltage_difference_normed'] = min(ovd_spec / ovd_den, 1.0)
    else:
        f_vals['output_voltage_difference_normed'] = 0.0

    # Keep stability as a bounded [0, 1] score using the configured range.
    stability_range = ckt.get('stability_norm', [0.0, 1.0])
    stability_lo = float(stability_range[0])
    stability_hi = float(stability_range[1])
    if stability_hi == stability_lo:
        stability_score = 0.0
    else:
        stability_score = 1.0 - abs((stability - stability_lo) / (stability_hi - stability_lo))
    f_vals['stability_normed'] = max(0.0, min(1.0, stability_score))

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

    merits = ckt['metrics_list']
    weights = [float(ckt.get(f'{m}_weight', 0.0)) for m in merits]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        weight_sum = 1.0

    fom = 0.0
    for merit, weight in zip(merits, weights):
        fom += (weight / weight_sum) * float(f_vals.get(f'{merit}_normed', 0.0))

    # Requirement: if both minimize-metrics meet spec, overall FOM is clipped to 1.
    if f_vals.get('q_curr_normed', 0.0) >= 1.0 and f_vals.get('output_voltage_difference_normed', 0.0) >= 1.0:
        fom = 1.0

    f_vals['fom'] = max(0.0, min(1.0, float(fom)))
    return f_vals