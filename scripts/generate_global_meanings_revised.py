#!/usr/bin/env python3
"""scripts/generate_global_meanings_revised.py

Revised global meanings generator with *no-alias by default* semantics.

Goal
- Avoid creating accidental aliases (over-merging distinct circuit concepts).
- Still deduplicate obvious formatting variants (case/whitespace/punctuation).
- Optionally collapse a small set of *true* synonym pairs that experienced
  designers would consider equivalent.

Writes (by default, to keep existing pipeline stable):
  - netlists/_performance_meanings.json
  - netlists/_substructures_ordered.json
  - netlists/_substructures_mapping.json

Also supports scanning either `comb_graph.json` or `<id>_comb_graph.json` per
circuit directory.

Usage:
  python3 scripts/generate_global_meanings_revised.py
  python3 scripts/generate_global_meanings_revised.py --netlists netlists --dry-run
  python3 scripts/generate_global_meanings_revised.py --out-prefix _revised

"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PREFERRED_PERF = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]


def _read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def find_comb_graph_path(circ_dir: Path) -> Optional[Path]:
    """Return the comb graph path for this circuit dir, or None."""
    primary = circ_dir / "comb_graph.json"
    if primary.exists():
        return primary
    alt = circ_dir / f"{circ_dir.name}_comb_graph.json"
    if alt.exists():
        return alt
    return None


def extract_base_metric(name: str) -> str:
    if not name:
        return name
    suffixes = ["-trade-off", "-ambiguous", "-directly-proportional", "-inversely-proportional"]
    for suf in suffixes:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def canonicalize_perf_name(name: str) -> str:
    """Canonicalize *formatting* only, then apply a small synonym set."""
    if not name:
        return name

    s = str(name).strip()

    # Formatting normalization
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s*[-/]+\s*", " ", s)  # keep words, drop separators
    s = s.strip()

    # Lowercase for matching
    sl = s.lower()

    # Preferred casing
    for p in PREFERRED_PERF:
        if sl == p.lower():
            return p

    # Designer-friendly true synonym collapses (KEEP THIS SHORT)
    perf_synonyms = {
        "dc gain": "Gain",
        "gain": "Gain",
        "ugb": "UGF",
        "unity gain frequency": "UGF",
        "unity gain bandwidth": "UGF",
        "propagation delay": "Delay",
        "input offset": "Offset",
        "psrr": "PSRR",
        "input referred noise": "input-referred noise",
        "inputreferrednoise": "input-referred noise",
        "kickbacknoise": "kickback noise",
        "outputswing": "output swing",
    }

    if sl in perf_synonyms:
        return perf_synonyms[sl]

    # Otherwise keep a stable lowercase form (but preserve hyphenation of common terms)
    # We keep "input-referred noise" as a special cased token.
    if sl == "input referred noise":
        return "input-referred noise"

    return sl


def canonicalize_substructure_name(raw: str) -> str:
    """Canonicalize substructure names with minimal aliasing.

    Strategy:
    - Normalize whitespace, punctuation, and obvious noise tokens.
    - Remove net-specific suffixes (e.g., "around net17/net22") because these
      are labeling artifacts and not distinct circuit concepts.
    - Avoid over-merging (e.g., do NOT merge "load resistors" with
      "resistive loads" or "output drivers" with "output stage").
    """
    if raw is None:
        return "unknown"

    s = str(raw).strip().lower()

    # Remove leading/trailing punctuation
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)

    # Normalize separators
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)

    # Remove common annotation artifacts
    s = re.sub(r"\bdev:[^\s]+\b", "", s)

    # Remove explicit net lists and net-specific phrases
    # Examples:
    # - "... around net17/net22/net19"
    # - "... into net25"
    # - "... between net07 and vout1"
    s = re.sub(r"\b(around|into|between)\s+net[0-9a-z_/]+(\s+and\s+[a-z0-9_/]+)?\b", "", s)
    s = re.sub(r"\bnet\d+\b", "", s)

    # Remove circuit instance tags like m1/m12/q3/r17/c5 when present as standalone tokens
    s = re.sub(r"\b[mqrc]\d+\b", "", s)

    # Collapse excess whitespace again
    s = re.sub(r"\s+", " ", s).strip()

    # Very small set of true synonyms (strict)
    # We only merge when the naming difference is purely stylistic.
    strict_synonyms = {
        "nmos cross coupled pair": "nmos cross coupled / regenerative pair",
        "nmos cross coupled / regenerative pair": "nmos cross coupled / regenerative pair",
        "precharge/keeper pmos loads for outputs": "precharge pmos",
    }
    if s in strict_synonyms:
        return strict_synonyms[s]

    return s if s else "unknown"


def iter_circuit_dirs(netlists_root: Path) -> Iterable[Tuple[str, Path]]:
    for fam_dir in sorted(p for p in netlists_root.iterdir() if p.is_dir()):
        fam = fam_dir.name
        # skip hidden/metadata dirs
        for circ in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            yield fam, circ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlists", default="netlists", help="Netlists root directory")
    ap.add_argument(
        "--out-prefix",
        default="",
        help="Optional prefix to write separate outputs, e.g. '_revised' -> netlists/_revised_substructures_ordered.json",
    )
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write files")
    args = ap.parse_args()

    netlists_root = Path(args.netlists)
    if not netlists_root.exists():
        raise FileNotFoundError(f"netlists directory not found: {netlists_root}")

    perf_counter: Counter[str] = Counter()
    sub_counter: Counter[str] = Counter()

    perf_raw_examples: Dict[str, List[str]] = defaultdict(list)
    sub_raw_examples: Dict[str, List[str]] = defaultdict(list)

    for fam, circ in iter_circuit_dirs(netlists_root):
        comb_path = find_comb_graph_path(circ)
        if comb_path is None:
            continue
        data = _read_json(comb_path)
        if not data:
            continue

        for node in data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            ntype = (node.get("type") or node.get("node_type") or "").lower()
            nid = node.get("id") or node.get("name") or ""
            if not nid:
                continue

            if ntype == "performance":
                base = extract_base_metric(str(nid))
                canon = canonicalize_perf_name(base)
                perf_counter[canon] += 1
                if len(perf_raw_examples[canon]) < 25 and str(nid) not in perf_raw_examples[canon]:
                    perf_raw_examples[canon].append(str(nid))
            elif ntype == "substructure":
                canon = canonicalize_substructure_name(str(nid))
                sub_counter[canon] += 1
                if len(sub_raw_examples[canon]) < 25 and str(nid) not in sub_raw_examples[canon]:
                    sub_raw_examples[canon].append(str(nid))

    # Order performance: preferred first (if present), then others sorted for stability
    perf_names = list(perf_counter.keys())
    ordered_perf: List[str] = []
    perf_lower_to_name = {n.lower(): n for n in perf_names}
    for p in PREFERRED_PERF:
        if p.lower() in perf_lower_to_name:
            ordered_perf.append(perf_lower_to_name[p.lower()])
    for n in sorted(perf_names):
        if n not in ordered_perf:
            ordered_perf.append(n)

    ordered_subs = [name for name, _ in sub_counter.most_common()]

    out_perf = netlists_root / f"{args.out_prefix}_performance_meanings.json".replace("__", "_")
    out_subs = netlists_root / f"{args.out_prefix}_substructures_ordered.json".replace("__", "_")
    out_map = netlists_root / f"{args.out_prefix}_meaning_mapping.json".replace("__", "_")

    mapping = {
        "performance": perf_raw_examples,
        "substructure": sub_raw_examples,
    }

    if args.dry_run:
        print(f"Would write: {out_perf}")
        print(f"Would write: {out_subs}")
        print(f"Would write: {out_map}")
        print(f"Performance meanings: {len(ordered_perf)}")
        print(f"Substructures: {len(ordered_subs)}")
        return

    _write_json(out_perf, {"performance_meanings": ordered_perf})
    _write_json(out_subs, {"ordered_substructures": ordered_subs})
    _write_json(out_map, mapping)

    print(f"Wrote performance meanings to: {out_perf} ({len(ordered_perf)})")
    print(f"Wrote substructure ordering to: {out_subs} ({len(ordered_subs)})")
    print(f"Wrote mapping to: {out_map}")


if __name__ == "__main__":
    main()
