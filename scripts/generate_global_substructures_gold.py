#!/usr/bin/env python3
"""scripts/generate_global_substructures_gold.py

Generate a hard-coded, designer-curated list of substructures ("gold vocabulary")
plus a deterministic mapping for every raw substructure string found in the
repo's comb graphs.

Design requirements (per user request)
- Provide a hard-coded ordered set of substructures.
- Provide a mapping from *raw* substructure names -> one gold substructure.
- Avoid punctuation artifacts, bullet prefixes, net-name tails, etc.
- Use an `unknown` bucket for anything unmapped.
- No implicit fuzzy aliasing: only explicit, readable rules.

Inputs
- Scans `netlists/<family>/<circuit>/comb_graph.json` OR
  `netlists/<family>/<circuit>/<circuit>_comb_graph.json`.

Outputs (by default, non-destructive filenames)
- `netlists/gold_substructures_ordered.json`
- `netlists/gold_substructures_mapping.json`

Usage
  python3 scripts/generate_global_substructures_gold.py
  python3 scripts/generate_global_substructures_gold.py --write-global

If `--write-global` is provided, it will overwrite:
- `netlists/_substructures_ordered.json`
- `netlists/_substructures_mapping.json`

"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NETLISTS_DEFAULT = "netlists"


GOLD_ORDERED_SUBSTRUCTURES: List[str] = [
    # Keep this ordered like how designers reason about blocks
    "differential pair",
    "input pair",
    "input stage",
    "tail current source",
    "current source",
    "current mirror",
    "active load",
    "load resistors",
    "resistive loads",
    "load capacitors",
    "coupling capacitors",
    "cross-coupling resistor",
    "emitter degeneration",
    "latch",
    "regeneration network",
    "cross-coupled pair",
    "cascode devices",
    "common-gate devices",
    "common-source devices",
    "transconductor",
    "common-mode sensing",
    "feedback/sensing network",
    "precharge network",
    "reset/discharge network",
    "clock/tail switch network",
    "clocked load network",
    "keeper/precharge loads",
    "bias network",
    "supply rails",
    "reference",
    "compensation capacitors",
    "decoupling capacitors",
    "output stage",
    "output driver",
    "output pull-up/pull-down",
    "output loads",
    "level shift",
    "common-mode feedback",
    "clamp/diode devices",
    "sampling/evaluate network",
    "input steering/injection",
    "control switch",
    "bjt gain/steering",
    "bjt output stage",
    "bjt preamp/tracking",
    "unknown",
]


@dataclass(frozen=True)
class RawRecord:
    raw: str
    count: int
    examples: Tuple[str, ...]


def find_comb_graph_path(circ_dir: Path) -> Optional[Path]:
    primary = circ_dir / "comb_graph.json"
    if primary.exists():
        return primary
    alt = circ_dir / f"{circ_dir.name}_comb_graph.json"
    if alt.exists():
        return alt
    return None


def iter_circuit_dirs(netlists_root: Path) -> Iterable[Tuple[str, Path]]:
    for fam_dir in sorted(p for p in netlists_root.iterdir() if p.is_dir()):
        if fam_dir.name.startswith("_"):
            continue
        for circ in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            yield fam_dir.name, circ


def sanitize_raw_substructure(s: str) -> str:
    """Remove formatting noise but keep the technical meaning."""
    if s is None:
        return ""

    x = str(s).strip()

    # Remove common list/bullet prefixes (sometimes repeated)
    x = re.sub(r"^(?:\s*[\-\*•]+\s*)+", "", x)

    # Normalize whitespace
    x = x.replace("\t", " ").replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()

    # Drop instance lists in parentheses if they are only device names (M1,M2,Q3,R4,C0,...)
    # Example: "Bias network (M6-M7-M8)" -> "Bias network"
    x = re.sub(r"\s*\((?:\s*[MQRC]\d+[\s,\-]*)+\)\s*$", "", x).strip()

    # Drop explicit net mentions (net17/net22, into net25, around net...) as they are dataset artifacts
    x = re.sub(r"\b(around|into|between)\s+net[0-9a-z_/]+(\s+and\s+[a-z0-9_/]+)?\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\bnet\d+\b", "", x, flags=re.IGNORECASE)

    # Remove leftover punctuation clusters and unmatched artifacts
    x = re.sub(r"[\(\)\[\]]", " ", x)
    x = re.sub(r"\s*[,]+\s*", " ", x)
    x = re.sub(r"\s*[\/]+\s*", " / ", x)  # keep explicit '/'
    x = re.sub(r"\s+", " ", x).strip()

    return x


def map_to_gold_bucket(sanitized: str) -> str:
    """Map a sanitized substructure string to one gold bucket.

    Rules are explicit and conservative. If not matched, return 'unknown'.
    """
    if not sanitized:
        return "unknown"

    s = sanitized.lower()

    # Helper: normalize common wording
    s = s.replace("cross coupled", "cross-coupled")
    s = re.sub(r"\s+", " ", s).strip()

    # Capacitors first (fairly unambiguous)
    if "load capac" in s:
        return "load capacitors"
    if "coupling capacitor" in s or "feedthrough cap" in s:
        return "coupling capacitors"
    if "coupling" in s and "capac" in s:
        return "coupling capacitors"

    # BJT-specific concepts
    if "emitter degeneration" in s or "emitter resistor" in s:
        return "emitter degeneration"

    # Differential / input
    if "differential pair" in s:
        return "differential pair"
    if "input quad" in s or "input quad" in s or "vin1-4" in s or "vin1" in s and "vin4" in s:
        return "sampling/evaluate network"
    if "input" in s and "pair" in s:
        return "input pair"
    if "input stage" in s:
        return "input stage"
    if "input / control" in s or "input control" in s:
        return "input stage"
    if "input compare branch" in s:
        return "sampling/evaluate network"

    # Tail/current sources
    if "tail" in s and ("current source" in s or "current" in s):
        return "tail current source"
    if "current source" in s or "current sink" in s:
        return "current source"

    # Mirrors / loads
    if "current mirror" in s:
        return "current mirror"
    if "current-mirror" in s:
        return "current mirror"
    if "active load" in s:
        return "active load"
    if "load resist" in s or "collector load" in s:
        return "load resistors"
    if "resistive load" in s:
        return "resistive loads"

    if "cross-coupling resistor" in s or ("cross" in s and "resistor" in s):
        return "cross-coupling resistor"

    # Resistive coupling / degeneration wording variants
    if "emitter coupling resistor" in s:
        return "emitter degeneration"
    if "interconnect" in s and "degeneration" in s:
        return "emitter degeneration"
    if s.strip() == "resistive coupling":
        return "resistive loads"

    # Latch/regeneration
    if "latch" in s:
        return "latch"
    if "regeneration" in s or "regenerative" in s:
        return "regeneration network"
    if "cross-coupled" in s:
        return "cross-coupled pair"

    # Cascode/common gate/common source/transconductor
    if "cascode" in s:
        return "cascode devices"
    if "common-gate" in s or "common gate" in s:
        return "common-gate devices"
    if "common-source" in s or "common source" in s:
        return "common-source devices"
    if "transconductor" in s or "gm" in s:
        return "transconductor"
    if "common-mode sensing" in s:
        return "common-mode sensing"
    if "sensing" in s or "feedback" in s:
        return "feedback/sensing network"

    # Explicit device group naming that implies a function
    if "pmos load devices" in s:
        return "active load"
    if "pmos current-source loads" in s:
        return "current source"
    if "nmos sink" in s and "output" in s:
        return "output stage"
    if "pmos output pair" in s:
        return "output stage"

    # Clocked/precharge/reset
    if "precharge" in s:
        return "precharge network"
    if "reset" in s or "discharge" in s:
        return "reset/discharge network"
    if "clock" in s and ("tail" in s or "switch" in s or "evaluate" in s or "gating" in s):
        return "clock/tail switch network"
    if "clocked load network" in s or ("clocked" in s and "load" in s):
        return "clocked load network"
    if "keeper" in s and ("load" in s or "precharge" in s):
        return "keeper/precharge loads"

    # Bias/reference
    if "bias" in s:
        return "bias network"
    if "reference" in s:
        return "reference"
    if "supply rails" in s or "vdd-vss" in s or "vdd" in s and "vss" in s:
        return "supply rails"

    # Capacitors
    if "compensation" in s or "rc load" in s:
        return "compensation capacitors"
    if "decap" in s or "decoupling" in s or "feedthrough cap" in s:
        return "decoupling capacitors"

    # Output
    if "cmfb" in s or "common mode feedback" in s or "common-mode feedback" in s:
        return "common-mode feedback"
    if "level shift" in s or "level-shift" in s:
        return "level shift"
    if "output driver" in s or "logic output driver" in s:
        return "output driver"
    if "output stage" in s:
        return "output stage"
    if "output inverter" in s:
        return "output driver"
    if "output pull" in s or ("pull-up" in s or "pull-down" in s):
        return "output pull-up/pull-down"
    if "output pmos loads" in s or "output loads" in s:
        return "output loads"
    if "output nmos devices" in s or "output pmos" in s:
        return "output stage"

    # Clamps/diodes
    if "clamp" in s or "diode" in s:
        return "clamp/diode devices"

    # Sampling
    if "sampling" in s or "evaluate" in s or "comparison" in s:
        return "sampling/evaluate network"

    # Steering/injection
    if "steering" in s or "inject" in s or "injection" in s:
        return "input steering/injection"

    # Control switches
    if "control switch" in s:
        return "control switch"

    # BJT buckets
    if "bjt" in s:
        # bjt pairs are still differential pairs conceptually
        if "differential" in s and "pair" in s:
            return "differential pair"
        if "output" in s:
            return "bjt output stage"
        if "preamp" in s or "tracking" in s:
            return "bjt preamp/tracking"
        if "gain" in s or "steering" in s:
            return "bjt gain/steering"

    return "unknown"


def collect_raw_substructures(netlists_root: Path) -> List[RawRecord]:
    raw_counter: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for fam, circ in iter_circuit_dirs(netlists_root):
        comb = find_comb_graph_path(circ)
        if comb is None:
            continue
        try:
            with comb.open("r") as f:
                data = json.load(f)
        except Exception:
            continue

        for node in data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            ntype = (node.get("type") or node.get("node_type") or "").lower()
            if ntype != "substructure":
                continue
            raw = node.get("id") or node.get("name") or ""
            if not raw:
                continue
            raw = str(raw)
            raw_counter[raw] += 1
            if len(examples[raw]) < 5:
                examples[raw].append(f"{fam}/{circ.name}")

    recs: List[RawRecord] = []
    for raw, cnt in raw_counter.most_common():
        recs.append(RawRecord(raw=raw, count=cnt, examples=tuple(examples[raw])))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlists", default=NETLISTS_DEFAULT)
    ap.add_argument("--write-global", action="store_true", help="Overwrite netlists/_substructures_*.json")
    args = ap.parse_args()

    netlists_root = Path(args.netlists)
    if not netlists_root.exists():
        raise FileNotFoundError(f"Netlists root not found: {netlists_root}")

    raw_records = collect_raw_substructures(netlists_root)

    # Build mapping: raw -> gold
    mapping: Dict[str, Dict] = {}
    bucket_counts: Counter[str] = Counter()

    for rec in raw_records:
        sanitized = sanitize_raw_substructure(rec.raw)
        gold = map_to_gold_bucket(sanitized)
        bucket_counts[gold] += rec.count
        mapping[rec.raw] = {
            "sanitized": sanitized,
            "gold": gold,
            "count": rec.count,
            "examples": list(rec.examples),
        }

    ordered = list(GOLD_ORDERED_SUBSTRUCTURES)
    if "unknown" not in ordered:
        ordered.append("unknown")

    ordered_obj = {"ordered_substructures": ordered}
    mapping_obj = {
        "gold_vocab": ordered,
        "bucket_counts": dict(bucket_counts),
        "mapping": mapping,
    }

    if args.write_global:
        out_ordered = netlists_root / "_substructures_ordered.json"
        out_mapping = netlists_root / "_substructures_mapping.json"
    else:
        out_ordered = netlists_root / "gold_substructures_ordered.json"
        out_mapping = netlists_root / "gold_substructures_mapping.json"

    out_ordered.write_text(json.dumps(ordered_obj, indent=2))
    out_mapping.write_text(json.dumps(mapping_obj, indent=2))

    print(f"Raw unique substructures: {len(raw_records)}")
    print(f"Wrote gold ordered list: {out_ordered}")
    print(f"Wrote raw->gold mapping: {out_mapping}")
    print("Bucket counts (by occurrences):")
    for k, v in bucket_counts.most_common():
        print(f"  {k:28s}  {v}")


if __name__ == "__main__":
    main()
