def read_work_region(lines, mosfet_str='mosfets'):
    """
    Read the work region of the MOSFET from the spice simulation results.
    Args:
        lines: list of strings, each string is a line in the spice simulation results.
        mosfet: string, the name of the MOSFET.
    Returns:
        work_region: string, the work region of the MOSFET.
    """
    start_idx = 0
    work_regions = {}
    for i, line in enumerate(lines):
        if mosfet_str in line:
            start_idx = i
            break
    for i, line in enumerate(lines[start_idx:]):
        if (('subckt' in line) and (len(line.split()) == 2 )):
            #print(lines[start_idx+i])
            #print(lines[start_idx+i+3])
            ckt_name = line.split()[1]
            work_region = lines[start_idx+i+3].split()[1]
            if work_region != 'Saturation':
                work_regions[ckt_name] = work_region
            
    return work_regions