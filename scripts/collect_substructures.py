#!/usr/bin/env python3
"""Scan comb_graph.json files for a family, normalize substructure names, and save ordered list.

Usage:
  python3 scripts/collect_substructures.py --family diff_amps --out netlists/diff_amps/_substructures_ordered.json

This script collects raw `substructure` nodes, applies normalization rules, and writes a canonical ordered list.
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter


def normalize_substructure_name(name: str, family: str = None) -> str:
    s = name.lower().strip()
    # remove leading/trailing punctuation
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    # replace common separators with space
    s = re.sub(r"[_\-]+", " ", s)
    # remove device/id tokens like m0, m1, m0-m1, r0, r1
    s = re.sub(r"\b[mr]\d+(?:-?[mr]?\d+)?\b", "", s)
    s = re.sub(r"\bdev:\w+\b", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Mapping rules (ordered)
    rules = [
        (r"\btail.*current\b", "tail current source"),
        (r"\bcurrent mirror\b", "current mirror"),
        (r"\bactive load\b", "active load"),
        (r"\bactive load current mirror\b", "active load"),
        (r"\bdifferential\b", "differential pair"),
        (r"\bdifferential pair\b", "differential pair"),
        (r"\bload resistor\b", "load resistors"),
        (r"\bload resistors\b", "load resistors"),
        (r"\binterconnection resistors\b", "load resistors"),
        (r"\bresistor\b", "load resistors"),
        (r"\bcurrent source\b", "current source"),
            (r"\bpmos.*active load\b", "active load"),
            (r"\bpmos active loads\b", "active load"),
            (r"\btail.*bias\b", "bias"),
            (r"\btail bias at ib1\b", "bias"),
    ]

    # Family-specific augmentation: add comparator rules when processing comparator family
    if family and family.lower() in ("comparators", "comparator"):
        comp_rules = [
            (r"\bpre-?amp\b", "preamp"),
            (r"\binput pair\b", "input pair"),
            (r"\binput stage\b", "input pair"),
            (r"\bregeneration\b", "regenerative latch"),
            (r"\bregenerative latch\b", "regenerative latch"),
            (r"\blatch\b", "latch"),
            (r"\bhysteresis\b", "hysteresis"),
            (r"\breference\b", "reference"),
            (r"\boutput stage\b", "output stage"),
            (r"\binput\s+transistor\b", "input transistor"),
            (r"\bpmos\b", "pmos transistor"),
            (r"\bnmos\b", "nmos transistor"),
        ]
        # insert comparator-specific rules before fallbacks
        rules = comp_rules + rules

    for patt, canon in rules:
        if re.search(patt, s):
            return canon

    # Fallback: remove numeric words and return cleaned text
    s = re.sub(r"\b\d+\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "":
        return "unknown"
    return s


def collect_raw_substructures(root: Path, family: str) -> Counter:
    counter = Counter()
    fam_dir = root / family
    if not fam_dir.exists():
        raise FileNotFoundError(f"Family dir not found: {fam_dir}")

    for circuit_dir in sorted(fam_dir.iterdir()):
        if not circuit_dir.is_dir():
            continue
        comb_graph = circuit_dir / 'comb_graph.json'
        if not comb_graph.exists():
            continue
        try:
            data = json.load(open(comb_graph))
        except Exception:
            continue
        nodes = data.get('nodes', [])
        for n in nodes:
            ntype = (n.get('type') or n.get('node_type') or '').lower()
            if ntype in ('substructure', 'sub-structure', 'sub-structure'):
                raw = n.get('id') or n.get('name') or ''
                raw = str(raw).strip()
                if raw:
                    counter[raw] += 1
    return counter


def collect_raw_performances(root: Path, family: str) -> Counter:
    counter = Counter()
    fam_dir = root / family
    if not fam_dir.exists():
        raise FileNotFoundError(f"Family dir not found: {fam_dir}")

    for circuit_dir in sorted(fam_dir.iterdir()):
        if not circuit_dir.is_dir():
            continue
        comb_graph = circuit_dir / 'comb_graph.json'
        if not comb_graph.exists():
            continue
        try:
            data = json.load(open(comb_graph))
        except Exception:
            continue
        nodes = data.get('nodes', [])
        for n in nodes:
            ntype = (n.get('type') or n.get('node_type') or '').lower()
            if ntype == 'performance':
                raw = n.get('id') or n.get('name') or ''
                raw = str(raw).strip()
                if raw:
                    counter[raw] += 1
    return counter


def build_performance_list(counter: Counter) -> tuple:
    # Extract base metrics: remove suffix variants like -trade-off, -ambiguous, -directly-proportional, -inversely-proportional
    def extract_base_metric(name: str) -> str:
        suffixes = ["-trade-off", "-ambiguous", "-directly-proportional", "-inversely-proportional"]
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    # Preferred ordering for known metrics
    preferred = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]
    names = list(counter.keys())
    # Extract base metrics only (filter variants)
    base_metrics = set()
    for n in names:
        base = extract_base_metric(n)
        base_metrics.add(base)

    # Preserve original casing from keys, but order by preferred first
    ordered = []
    lower_to_name = {n.lower(): n for n in base_metrics}
    for p in preferred:
        if p.lower() in lower_to_name:
            ordered.append(lower_to_name[p.lower()])
    # append other base metrics not already included
    for n in sorted(base_metrics):
        if n not in ordered:
            ordered.append(n)
    mapping = {n: [n] for n in ordered}
    return ordered, mapping


def build_canonical_list(counter: Counter, family: str = None) -> tuple:
    # Normalize all names and count canonical occurrences
    canon_counter = Counter()
    mapping = {}
    for raw, cnt in counter.items():
        canon = normalize_substructure_name(raw, family=family)
        canon_counter[canon] += cnt
        mapping.setdefault(canon, set()).add(raw)

    # Sort canonical names by frequency (descending), then alphabetically
    ordered = sorted(canon_counter.items(), key=lambda x: (-x[1], x[0]))
    return [name for name, _ in ordered], mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--root', default='netlists')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    root = Path(args.root)
    family = args.family

    # collect substructures
    raw_counter = collect_raw_substructures(root, family)
    if not raw_counter:
        print(f"No substructures found for family {family}")
    else:
        canonical_list, mapping = build_canonical_list(raw_counter, family=family)
        # Default output path for substructures
        out_path = Path(args.out) if args.out else root / family / '_substructures_ordered.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'ordered_substructures': canonical_list, 'mapping': {k: list(v) for k, v in mapping.items()}}, f, indent=2)
        print(f"Wrote canonical substructure list to: {out_path}")
        print("Canonical substructures (ordered):")
        for i, name in enumerate(canonical_list, 1):
            raw_examples = ', '.join(list(mapping[name])[:3])
            print(f"{i:2d}. {name}  (raw examples: {raw_examples})")

    # collect performance meanings for family and write _performance_meanings.json
    perf_counter = collect_raw_performances(root, family)
    if not perf_counter:
        print(f"No performance nodes found for family {family}")
    else:
        perf_list, perf_mapping = build_performance_list(perf_counter)
        perf_out = Path(args.out).parent / '_performance_meanings.json' if args.out else root / family / '_performance_meanings.json'
        perf_out.parent.mkdir(parents=True, exist_ok=True)
        with open(perf_out, 'w') as f:
            json.dump({'performance_meanings': perf_list, 'mapping': {k: list(v) for k, v in perf_mapping.items()}}, f, indent=2)
        print(f"Wrote performance meanings to: {perf_out}")
        print("Performance meanings (ordered):")
        for i, name in enumerate(perf_list, 1):
            print(f"{i:2d}. {name}")

if __name__ == '__main__':
    main()
