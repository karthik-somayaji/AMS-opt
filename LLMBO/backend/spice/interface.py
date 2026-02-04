import os
import sys
import time
import subprocess

# This module is imported via `backend.spice.interface` after `LLMBO/core/task.py`
# mutates `sys.path`. Use intra-package relative imports so it works both when
# `LLMBO` is run as a script (not a package) and when imported as a package.
from .objective_new import objective  # noqa: E402
from .objective import read_results  # noqa: E402
from .aux_info import read_work_region  # noqa: E402


def hspice_eval_f(point_to_evaluate, args):
    '''
    Evaluate a single point on hspice simulation
    Args: point_to_evaluate (dict), string containing point to be evaluated
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
            f.write(point_to_evaluate)

    ckt_name = args['ckt_name']
    ckt_dir = args['ckt_dir']

    # HSPICE simulation
    cmd = f'hspice {ckt_dir}/{ckt_name}.sp -o {ckt_dir}/{ckt_name}.lis'
    # run command, wait for it to finish
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    # read results from {ckt_name}.ma0, acm is the first value in the fourth line
    with open(f'{ckt_dir}/{ckt_name}.ma0') as f:
        lines = f.readlines()
        acm = read_results(lines[3].split()[0], 'acm', args)
    # read results from {ckt_name}.ma1, gain, gbw, pm is the 1st, 2nd, 3rd values in the fifth line
    with open(f'{ckt_dir}/{ckt_name}.ma1') as f:
        lines = f.readlines()
        gain = read_results(lines[4].split()[0], 'gain', args)
        gbw = read_results(lines[4].split()[1], 'gbw', args)
        pm = read_results(lines[4].split()[2], 'pm', args)
        cmrr = gain - acm
    # read results from {ckt_name}.dp0
    with open(f'{ckt_dir}/{ckt_name}.dp0') as f:
        lines = f.readlines()
        pow_vol = read_results(lines[25].split()[1], 'pow_vol', args)
        pow_cur = read_results(lines[49].split()[1], 'pow_cur', args)
        pow = pow_vol + pow_cur
        work_regions = read_work_region(lines)
    # update fvals
    fvals['gbw'] = gbw
    fvals['gain'] = gain
    fvals['cmrr'] = cmrr
    fvals['pm'] = pm
    fvals['pow_vol'] = pow_vol
    fvals['pow_cur'] = pow_cur
    fvals['pow'] = pow
    fvals['acm'] = acm
    fvals['work_regions'] = work_regions
    fvals['aux_info'] += f'The following mosfets are not in saturation region:\n '
    # fvals['metrics'] = {'gbw': gbw, 'gain': gain, 'cmrr': cmrr, 'pm': pm, 'pow': pow}
    # append (mosfet, work_region) to aux_info
    for mosfet, work_region in work_regions.items():
        fvals['aux_info'] += f'{mosfet}: {work_region}\n '
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
    point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=4p r1=800'
    #point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u c=4p r=800'
    #point_to_evaluate = '.param w1=120u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u c=4p r=800'



    args = {
        'ckt_dir': '/home/karthik/sim_clean/LLMBO/circuits/amp2',
        'ckt_name': 'amp2_0',
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
    point_to_evaluate, fvals = hspice_eval_f(point_to_evaluate, args)
    print(f'point_to_evaluate: {point_to_evaluate}')
    print(f'fvals: {fvals}')

    print(f"gain={fvals['gain']:.3f} cmrr={fvals['cmrr']:.3f} gbw={fvals['gbw']:.3f} pm={fvals['pm']:.3f}")