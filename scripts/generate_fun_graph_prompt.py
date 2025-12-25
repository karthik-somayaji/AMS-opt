#!/usr/bin/env python3
"""
Generate the initial functional-graph prompt by inserting a netlist into the template
and writing the resulting prompt to the circuit directory.

Usage examples:
  python scripts/generate_fun_graph_prompt.py --circuit netlists/diff_amps/75/
  python scripts/generate_fun_graph_prompt.py --netlist netlists/diff_amps/75/75.cir --out netlists/diff_amps/75/graph_query_prompt.txt

If no --out is given the script will write `graph_query_prompt.txt` into the circuit directory.
"""
import argparse
import os
import glob
from typing import Optional


def find_netlist_in_dir(d: str) -> Optional[str]:
    patterns = ["*.cir", "*.sp", "*.net"]
    for p in patterns:
        matches = glob.glob(os.path.join(d, p))
        if matches:
            return matches[0]
    return None


PROMPT_TEMPLATE = """
Given a circuit netlist and a set of performance metrics, do the following and output everything together as plain tuple lists, without any explanations:

1. **Infer P-graph (performance–performance relations).**
   Output entries of the form:
   `(Metric_A, Relationship, Metric_B)`
   where `Relationship ∈ {{ 'trade-off', 'directly-proportional', 'ambiguous' }}`.

2. **Infer PS-graph (substructure–performance relations).**
   First, identify meaningful circuit substructures (e.g., “M0-M1 differential pair”, “M2-M3 active load current mirror”, “R0-R1 load resistors”).
   Then output entries of the form:
   `(Substructure, 'influences', Metric)`
   only if that substructure significantly affects the metric.

3. **Infer PSX-graph (parameter–substructure–performance relations).**
   Treat device sizes, bias currents, resistances, etc. as design parameters (e.g., `W_M0, L_M0, IB1, R0, R1, VB1`).
   Output:

   * `(Parameter, 'belongs-to', Substructure)` if the parameter structurally belongs to that substructure.
   * `(Parameter, Relationship, Metric)` where `Relationship ∈ {{ 'directly-proportional', 'inversely-proportional' }}`, indicating how increasing the parameter value affects the metric.

**Important formatting constraints:**

* Use exactly the tuple formats above.
* Use only the allowed relationship labels.
* Do **not** include any explanations, text, or headings—only the tuples.

Given the above instructions, construct the above tuples as entries of a knowledge graph. Specifically, output a json format where all nodes (nodes in P, PS and PSX graphs) are listed under the field 'nodes'.
For example:
"nodes": [
    {{"id": "Active load",              "type": "substructure"}},
    {{"id": "CMRR",                     "type": "performance"}},
    {{"id": "Differential pair",        "type": "substructure"}}, etc]
    
 All tuples in the graphs should be listed under a 'links' field as a list of dictionaries.
 For example: "links": [
    {{"source": "Gain", "target": "CMRR", "relation": "ambiguous"}}, etc]


Netlist:
```
{netlist}
```

"""


def build_prompt(netlist_text: str) -> str:
    return PROMPT_TEMPLATE.format(netlist=netlist_text)


def main():
    p = argparse.ArgumentParser(description="Insert netlist into functional-graph prompt template and write to circuit dir")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--circuit", help="Path to a circuit directory (looks for .cir/.sp files inside)")
    group.add_argument("--netlist", help="Path to a specific netlist file")
    p.add_argument("--out", help="Output file path (defaults to <circuit>/graph_query_prompt.txt)")
    args = p.parse_args()

    netlist_path = args.netlist
    if args.circuit:
        if not os.path.isdir(args.circuit):
            raise SystemExit(f"Circuit path is not a directory: {args.circuit}")
        netlist_path = find_netlist_in_dir(args.circuit)
        if not netlist_path:
            raise SystemExit(f"No netlist (.cir/.sp/.net) found in {args.circuit}")

    if not os.path.exists(netlist_path):
        raise SystemExit(f"Netlist file not found: {netlist_path}")

    with open(netlist_path, "r") as f:
        netlist_text = f.read().strip()

    prompt = build_prompt(netlist_text)

    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.dirname(netlist_path)
        out_path = os.path.join(out_dir, "graph_query_prompt.txt")

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(prompt)

    print(f"Wrote prompt to {out_path}")


if __name__ == "__main__":
    main()
