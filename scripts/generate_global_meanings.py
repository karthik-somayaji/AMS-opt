#!/usr/bin/env python3
"""
Generate global performance meanings and substructure ordering across all families.
Writes:
  - netlists/_performance_meanings.json
  - netlists/_substructures_ordered.json

This scans all `netlists/<family>/*/comb_graph.json` files.
"""
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

NETLISTS = "netlists"

PREFERRED_PERF = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]

DEFAULT_FAMILIES = ["comparators", "diff_amps", "LDO", "op-amp"]
DEFAULT_RULES_FILE = Path(__file__).with_name("meaning_aliases.json")


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_rules(path: Path | None) -> dict:
    if not path:
        return {"performance": {"canonical": {}, "regex": {}}, "substructure": {"canonical": {}, "regex": {}}}
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    data = json.loads(path.read_text())
    for section in ("performance", "substructure"):
        data.setdefault(section, {})
        data[section].setdefault("canonical", {})
        data[section].setdefault("regex", {})
    return data


def _compile_rules(rules: dict) -> dict:
    compiled = {"performance": {"canonical": {}, "regex": []}, "substructure": {"canonical": {}, "regex": []}}
    for section in ("performance", "substructure"):
        # canonical exact-match map (case/space insensitive)
        exact = {}
        for canon, patterns in rules.get(section, {}).get("canonical", {}).items():
            for p in patterns:
                exact[_norm_key(p)] = canon
        compiled[section]["canonical"] = exact

        # regex rules, kept ordered (first match wins)
        rx = []
        for canon, patterns in rules.get(section, {}).get("regex", {}).items():
            for p in patterns:
                rx.append((re.compile(p, flags=re.IGNORECASE), canon))
        compiled[section]["regex"] = rx
    return compiled


def apply_manual_rules(section: str, raw_name: str, compiled_rules: dict) -> str | None:
    if not raw_name:
        return None
    key = _norm_key(raw_name)
    canon_exact = compiled_rules.get(section, {}).get("canonical", {}).get(key)
    if canon_exact:
        return canon_exact
    for patt, canon in compiled_rules.get(section, {}).get("regex", []):
        if patt.search(raw_name) or patt.search(key):
            return canon
    return None


def normalize_substructure_name(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\b[mr]\d+(?:-?[mr]?\d+)?\b", "", s)
    s = re.sub(r"\bdev:\w+\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Drop long descriptive tails like "(..." to reduce noisy near-duplicates.
    # We keep the prefix before '(' as the core concept.
    s = re.sub(r"\s*\(.*$", "", s).strip()

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

    # Medium (B) reduction: collapse messy aliases into canonical buckets.
    # Keep this conservative: merge only cases that are clearly variations.
    alias_rules = [
        (r"\bmirror\b", "current mirror"),

        # Resistive loads family
        (r"\bload resistors?\b", "load resistors"),
        (r"\boutput resistors?\b", "load resistors"),
        (r"\bresistive loads?\b", "resistive loads"),
        (r"\bresistive load network\b", "resistive loads"),
        (r"\bresistive loads? to vdd\b", "resistive loads"),
        (r"\bresistive load / sensing network\b", "resistive loads"),

        # Capacitors
        (r"\bload capacitors?\b", "load capacitors"),
        (r"\bcompensation/load capacitor\b", "compensation/load capacitor"),
        (r"\bcompensation/feedback capacitors?\b", "compensation/feedback capacitors"),
        (r"\bcompensation/cross coupling capacitors?\b", "compensation/cross coupling capacitors"),

        # Supplies
        (r"\bvdd vss supply rails\b", "supply rails"),

        # Degeneration
        (r"\bq\d+ emitter degeneration\b", "emitter degeneration"),

        # Clock / precharge
        (r"\bclocked precharge nmos\b", "clocked precharge nmos"),
    ]

    for patt, canon in rules:
        if re.search(patt, s):
            return canon

    for patt, canon in synonym_rules:
        if re.search(patt, s):
            return canon

    for patt, canon in alias_rules:
        if re.search(patt, s):
            return canon

    s = re.sub(r"\b\d+\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)

    # Additional targeted collapses based on prefixes after cleanup
    if s.startswith("pmos load devices"):
        return "pmos load devices"
    if s.startswith("pmos output device"):
        return "pmos output devices"
    if s.startswith("nmos common gate/cascode output devices"):
        return "nmos cascode/common gate output devices"
    if s.startswith("nmos output devices"):
        return "nmos output devices"
    if s.startswith("pmos input quad"):
        return "pmos input quad"

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


def normalize_perf_name(name: str) -> str:
    if not name:
        return name
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ").replace("/", " ")
    s = s.strip()
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate global meanings across selected netlist families")
    p.add_argument(
        "--netlists-root",
        default=NETLISTS,
        help="Netlists root directory (default: netlists)",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        help="Families to include (default: comparators diff_amps LDO op-amp)",
    )
    p.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_FILE),
        help=f"Alias rules JSON file (default: {DEFAULT_RULES_FILE.name})",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only keep canonical names seen at least this many times",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write JSON outputs; print summary only",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Optional path to write a curation report JSON (top raw->canonical mappings + counts)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.netlists_root)
    if not root.exists():
        raise FileNotFoundError("netlists directory not found")

    rules_path = Path(args.rules) if args.rules else None
    rules = load_rules(rules_path)
    compiled_rules = _compile_rules(rules)

    sub_counter = Counter()
    perf_counter = Counter()

    # For reversibility/debugging:
    # - canonical -> raw examples
    # - raw -> canonical (with counts)
    sub_raw_examples = defaultdict(list)
    sub_raw_to_canon_count = defaultdict(Counter)
    perf_raw_to_canon_count = defaultdict(Counter)

    # Walk selected families and circuit dirs
    families = []
    for f in args.families:
        fam_path = root / f
        if fam_path.is_dir():
            families.append(fam_path)
    if not families:
        raise FileNotFoundError(f"No valid families found under {root}: {args.families}")

    for fam in sorted(families, key=lambda p: p.name.lower()):
        for circ in sorted(fam.iterdir()):
            if not circ.is_dir():
                continue
            # Support both common filenames:
            # - comb_graph.json
            # - <id>_comb_graph.json
            # Prefer comb_graph.json when both exist.
            comb = circ / "comb_graph.json"
            if not comb.exists():
                circ_id = circ.name
                alt = circ / f"{circ_id}_comb_graph.json"
                if alt.exists():
                    comb = alt
                else:
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
                    manual = apply_manual_rules("substructure", nid, compiled_rules)
                    canon = manual or normalize_substructure_name(nid)
                    sub_counter[canon] += 1
                    if nid not in sub_raw_examples[canon] and len(sub_raw_examples[canon]) < 20:
                        sub_raw_examples[canon].append(nid)
                    sub_raw_to_canon_count[nid][canon] += 1
                elif ntype == "performance":
                    base = normalize_perf_name(extract_base_metric(nid))
                    manual = apply_manual_rules("performance", base, compiled_rules)
                    canon = manual or canonicalize_perf_name(base)
                    perf_counter[canon] += 1
                    perf_raw_to_canon_count[nid][canon] += 1

    # Apply frequency cutoff
    sub_counter = Counter({k: v for k, v in sub_counter.items() if v >= args.min_count})
    perf_counter = Counter({k: v for k, v in perf_counter.items() if v >= args.min_count})

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
    out_subs_map = root / "_substructures_mapping.json"
    out_perf_map = root / "_performance_mapping.json"

    report = {
        "families": [p.name for p in families],
        "min_count": args.min_count,
        "counts": {
            "substructures": len(sub_counter),
            "performance": len(perf_counter),
        },
        "top_substructures": sub_counter.most_common(40),
        "top_performance": perf_counter.most_common(40),
        "raw_to_canonical": {
            "substructure": {raw: cnt.most_common(3) for raw, cnt in sub_raw_to_canon_count.items()},
            "performance": {raw: cnt.most_common(3) for raw, cnt in perf_raw_to_canon_count.items()},
        },
    }

    if not args.dry_run:
        out_perf.write_text(json.dumps({"performance_meanings": ordered_perf}, indent=2))
        out_subs.write_text(json.dumps({"ordered_substructures": ordered_subs}, indent=2))
        out_subs_map.write_text(json.dumps({"mapping": sub_raw_examples}, indent=2))
        out_perf_map.write_text(
            json.dumps(
                {
                    "mapping": {
                        canon: [] for canon in ordered_perf
                    }
                },
                indent=2,
            )
        )
        print(f"Wrote global performance meanings to: {out_perf}")
        print(f"Wrote global substructure ordering to: {out_subs}")
        print(f"Wrote global substructure mapping to: {out_subs_map}")
        print(f"Wrote global performance mapping to: {out_perf_map}")
    else:
        print("Dry run: not writing output JSON files")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"Wrote curation report to: {args.report}")

    print("Top performance:", ordered_perf[:20])
    print("Top substructures:", ordered_subs[:30])


if __name__ == '__main__':
    main()
