#!/usr/bin/env python3
"""
Generate global performance meanings and substructure ordering across all families.
Writes:
  - netlists/_performance_meanings.json
  - netlists/_substructures_ordered.json

This scans all `netlists/<family>/*/comb_graph.json` files.
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

NETLISTS = "netlists"

PREFERRED_PERF = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]


def normalize_substructure_name(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\b[mr]\d+(?:-?[mr]?\d+)?\b", "", s)
    s = re.sub(r"\bdev:\w+\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    rules = [
        (r"\btail.*current\b", "tail current source"),
        (r"\bcurrent mirror\b", "current mirror"),
        (r"\bactive load current mirror\b", "active load"),
        (r"\bpmos.*active load\b", "active load"),
        (r"\bpmos active loads\b", "active load"),
        (r"\bactive load\b", "active load"),
        (r"\btail.*bias\b", "bias"),
        (r"\btail bias at ib1\b", "bias"),
        (r"\bdifferential\b", "differential pair"),
        (r"\bload resistor\b", "load resistors"),
        (r"\bload resistors\b", "load resistors"),
        (r"\binterconnection resistors\b", "load resistors"),
        (r"\bresistor\b", "load resistors"),
        (r"\bcurrent source\b", "current source"),
        # comparator-specific
        (r"\bpre-?amp\b", "precharge pmos"),
        (r"\bregeneration\b", "latch"),
        (r"\blatch\b", "latch"),
        (r"\bhysteresis\b", "hysteresis"),
        (r"\breference\b", "reference"),
    ]

    # Additional synonym normalizations to collapse similar names
    synonym_rules = [
        (r"\bbias network\b", "bias"),
        (r"\bbias transistor\b", "bias"),
        (r"\bclocked tail network\b", "clock tail network"),
        (r"\bclocked tail\b", "clock tail network"),
        (r"\bclock tail network\b", "clock tail network"),
        (r"\bclocked nmos reset pair\b", "clocked nmos reset"),
        (r"\bclocked nmos reset\b", "clocked nmos reset"),
        (r"\breset nmos pair\b", "clocked nmos reset"),
        (r"\bprecharge pmos pair\b", "precharge pmos"),
        (r"\bprecharge pmos\b", "precharge pmos"),
        (r"\bregenerative latch\b", "latch"),
        (r"\bbias\b", "bias"),
    ]

    for patt, canon in rules:
        if re.search(patt, s):
            return canon

    for patt, canon in synonym_rules:
        if re.search(patt, s):
            return canon

    s = re.sub(r"\b\d+\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else "unknown"


def extract_base_metric(name: str) -> str:
    if not name:
        return name
    suffixes = ["-trade-off", "-ambiguous", "-directly-proportional", "-inversely-proportional"]
    for suf in suffixes:
        if name.endswith(suf):
            return name[:-len(suf)]
    return name


def canonicalize_perf_name(name: str) -> str:
    """Return canonical performance name (preferred casing if available), collapse case variants."""
    if not name:
        return name
    n = name.strip()
    # normalize whitespace and case
    nl = n.lower().strip()
    # preferred map
    for p in PREFERRED_PERF:
        if nl == p.lower():
            return p
    # otherwise return lower-case canonical
    return nl


def main():
    root = Path(NETLISTS)
    if not root.exists():
        raise FileNotFoundError("netlists directory not found")

    sub_counter = Counter()
    perf_counter = Counter()

    # Walk families and circuit dirs
    for fam in sorted(p for p in root.iterdir() if p.is_dir()):
        for circ in sorted(fam.iterdir()):
            if not circ.is_dir():
                continue
            comb = circ / "comb_graph.json"
            if not comb.exists():
                continue
            try:
                data = json.load(open(comb))
            except Exception:
                continue
            for n in data.get("nodes", []):
                ntype = (n.get("type") or n.get("node_type") or "").lower()
                nid = n.get("id") or n.get("name") or ""
                if not nid:
                    continue
                if ntype == "substructure":
                    canon = normalize_substructure_name(nid)
                    sub_counter[canon] += 1
                elif ntype == "performance":
                    base = extract_base_metric(nid)
                    canon = canonicalize_perf_name(base)
                    perf_counter[canon] += 1

    # Build ordered substructures by frequency
    ordered_subs = [name for name, _ in sub_counter.most_common()]

    # Build ordered performance meanings: preferred first then others
    perf_names = list(perf_counter.keys())
    ordered_perf = []
    # ensure preferred (with preferred casing) appear first if present
    perf_lower_to_name = {n.lower(): n for n in perf_names}
    for p in PREFERRED_PERF:
        if p.lower() in perf_lower_to_name:
            # use preferred casing
            ordered_perf.append(p)
    # append other detected perf names (use canonicalized names from counting)
    for n in sorted(perf_names):
        if n not in ordered_perf:
            ordered_perf.append(n)

    # Write global files
    out_perf = root / "_performance_meanings.json"
    out_subs = root / "_substructures_ordered.json"

    out_perf.write_text(json.dumps({"performance_meanings": ordered_perf}, indent=2))
    out_subs.write_text(json.dumps({"ordered_substructures": ordered_subs}, indent=2))

    print(f"Wrote global performance meanings to: {out_perf}")
    print(ordered_perf)
    print(f"Wrote global substructure ordering to: {out_subs}")
    print(ordered_subs[:30])


if __name__ == '__main__':
    main()
