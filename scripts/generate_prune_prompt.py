#!/usr/bin/env python3
"""
Generate a prompt asking an LLM to identify removable components from a netlist.

Usage examples:
  # point to a circuit directory (looks for .cir/.sp files)
  python scripts/generate_prune_prompt.py --circuit netlists/diff_amps/75/ --out netlists/diff_amps/75/prune_prompt.txt

  # point to a specific netlist file, print to stdout
  python scripts/generate_prune_prompt.py --netlist netlists/diff_amps/75/75.cir

The script writes a plain-text prompt by default. Use --jsonl to write a one-line JSONL
object with fields {"circuit_id","netlist_file","prompt"}.
"""
import argparse
import json
import os
import glob
from typing import Optional


def find_netlist_in_dir(d: str) -> Optional[str]:
    """Look for common netlist files (.cir, .sp) in a directory and return the first match."""
    patterns = ["*.cir", "*.sp", "*.net"]
    for p in patterns:
        matches = glob.glob(os.path.join(d, p))
        if matches:
            # return the first non-empty candidate
            return matches[0]
    return None


PROMPT_TEMPLATE = (
    "Given the following SPICE netlist of a differential amplifier, identify {n} components "
    "that can be deleted without significantly compromising the functionality of the circuit.\n\n"
    "For each component you identify, provide a short explanation (1-2 sentences) describing why "
    "removing it would have minimal impact and any caveats or dependencies to be aware of.\n\n"
    "Return the answer as a JSON array of objects with keys: 'component', 'reason', 'impact_estimate'.\n\n"
    "Netlist (file: {netlist_file}):\n {netlist}\n "

)


def build_prompt(netlist_text: str, netlist_file: str, n: int = 3) -> str:
    return PROMPT_TEMPLATE.format(n=n, netlist_file=os.path.basename(netlist_file), netlist=netlist_text)


def main():
    p = argparse.ArgumentParser(description="Generate a prune prompt from a netlist or circuit directory")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--circuit", help="Path to a circuit directory (looks for .cir/.sp files inside)")
    group.add_argument("--netlist", help="Path to a specific netlist file")
    p.add_argument("--out", help="Output file to write prompt (default: prints to stdout)")
    p.add_argument("--jsonl", action="store_true", help="Write prompt as a one-line JSONL object")
    p.add_argument("--n", type=int, default=3, help="Number of components to request (default: 3)")
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

    prompt = build_prompt(netlist_text, netlist_path, n=args.n)

    if args.jsonl:
        obj = {"circuit_id": os.path.basename(os.path.dirname(netlist_path)),
               "netlist_file": os.path.basename(netlist_path),
               "prompt": prompt}
        out_text = json.dumps(obj, ensure_ascii=False)
    else:
        out_text = prompt

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out_text)
        print(f"Wrote prompt to {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
