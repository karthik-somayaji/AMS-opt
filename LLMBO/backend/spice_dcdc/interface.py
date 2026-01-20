import os
import sys
import time
import subprocess
import numpy as np
sys.path.append("../")

from backend.spice_dcdc.objective import objective, read_results # noqa: E402
#from backend.spice_comp.aux_info import read_work_region # noqa: E402


def hspice_eval_f_dcdc(point_to_evaluate, args):
    '''
    Evaluate a single point on hspice simulation
    Args: point_to_evaluate (dict), dictionary containing point to be evaluated
            args (dict), dictionary containing circuit information
    Returns: (point_to_evaluate, f_vals) (dict, dict)
             point_to_evaluate (dict) is the point evaluated
             f_vals (dict) is a dictionary that can track an arbitrary number of metrics, but must contain 'score' which is what LLAMBO optimizer tries to optimize by default

    Example f_vals:
    f_vals = {
        'fom': float,                     -> 'fom' is what the LLAMBO optimizer tries to optimize
        ...
    }
    '''
    # init fvals: gbw, gain, cmrr, pm
    fvals = { 'fom': 0.0, 'gbw': 0.0, 'gain': 0.0, 'cmrr': 0.0, 'pm': 0.0, 'aux_info':'' }

    ckt_dir = args['ckt_dir']
    param_file = os.path.join(ckt_dir, 'param0.inc')

    # create file if it does not exist
    if not os.path.exists(param_file):
        open(param_file, 'w').close()
    with open(param_file, 'w') as f:
            print('point_to_ebaluate: ' , point_to_evaluate)
            f.write(point_to_evaluate)

    ckt_name = args['ckt_name']
    ckt_dir = args['ckt_dir']

    # HSPICE simulation
    cmd = f'hspice {ckt_dir}/{ckt_name}.sp -o {ckt_dir}/{ckt_name}.lis'
    # run command, wait for it to finish
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    # read results from {ckt_name}.ma0, acm is the first value in the fourth line
    # read results from {ckt_name}.ma1, gain, gbw, pm is the 1st, 2nd, 3rd values in the fifth line
    with open(f'{ckt_dir}/{ckt_name}.mt0') as f:
        lines = f.readlines()
        power_output_curr = read_results(lines[5].split()[1], 'power_output', args)
        stability = float(lines[4].split()[1]) - float(lines[4].split()[2])#read_results(lines[4].split()[1], 'ugf', args)
        output_voltage_difference = 0.6 - float(lines[4].split()[3])
    
    # update fvals
    fvals['power_output'] = np.abs(power_output_curr)
    fvals['stability'] = stability
    fvals['output_voltage_difference'] = output_voltage_difference
    # fvals['work_regions'] = work_regions
    # fvals['aux_info'] += f'The following mosfets are not in saturation region:\n '
    # for mosfet, work_region in work_regions.items():
    #     fvals['aux_info'] += f'{mosfet}: {work_region}\n '
    # calculate fom
    fvals = objective(fvals, args)

    return point_to_evaluate, fvals

# main function
if __name__ == '__main__':
    # a two-stage opamp example, 8 transistors, 1 resistor, 1 capacitor

    point_to_evaluate = ".param w1=25u l1=2.5u w2=20u l2=2.5u w3=25u l3=2u w4=20u l4=2u w5=10u l5=2u w6=25u l6=2u w7=30u l7=2.5u w8=2u l8=2u r1=20k c1=0.2p"
    point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=200u l1=1u w2=200u l2=1u w3=20u l3=1u w4=20u l4=1u w5=15u l5=1u w6=15u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=250u l1=1u w2=250u l2=1u w3=25u l3=1u w4=25u l4=1u w5=20u l5=1u w6=20u l6=1u w7=250u l7=1u w8=25u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=300u l1=1u w2=300u l2=1u w3=30u l3=1u w4=30u l4=1u w5=20u l5=1u w6=20u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=1.5k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=2k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=800'
    #point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=4p r1=800'
    point_to_evaluate = '.param r1 =  0.927224 r2 =  0.970933 l1 =  0.931941 l3 =  0.959592 l4 =  0.807255 l6 =  0.99506 l8 =  0.848993 l9 =  0.900406 \
     l10 = 0.936496 l13 = 0.985991 w15 = 0.978171 w1 =  0.882312 w4 =  0.973641 w6 =  0.914594 w8 =  0.863931 w9 =  0.925017 w10 = 0.914919 w13 = 0.913944'

    args = {
        'ckt_dir': '/home/karthik/sim_clean/LLMBO/circuits/FC',
        'ckt_name': 'fc_0',
        'gbw_norm': [1, 1e6],
        'gain_norm': [-20, 60],
        'cmrr_norm': [0, 80],
        'pm_norm': [0, 180],
        "pow_norm": [0, 1.5e-5],
        'gbw_weight': 0.25,
        'gain_weight': 0.25,
        'cmrr_weight': 0.25,
        'pm_weight': 0.25,
        "pow_weight": -0.25,
        'gbw_failed': 0.0,
        'gain_failed': 0.0,
        'pm_failed': 0.0,
        'acm_failed': 0.0,
    }
    point_to_evaluate, fvals = hspice_eval_f_comp(point_to_evaluate, args)
    print(f'point_to_evaluate: {point_to_evaluate}')
    print(f'fvals: {fvals}')

    print(f"gain={fvals['gain']:.3f} cmrr={fvals['cmrr']:.3f} gbw={fvals['gbw']:.3f} pm={fvals['pm']:.3f}")