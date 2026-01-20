#!/usr/bin/env python3
"""scripts/comb_graph_to_gnn_global.py

A wrapper/variant of `scripts/comb_graph_to_gnn.py` that forces a fixed, global
feature schema so *new* circuit folders (e.g. under `LLMBO/`) produce the same
feature dimension as the training netlists.

Why this exists
- `comb_graph_to_gnn.py` infers `netlists_root` from the input path.
- For inputs under `netlists/<family>/<id>/...`, it finds global meaning files
  like `netlists/_performance_meanings.json` and `netlists/_substructures_ordered.json`.
- For inputs under `LLMBO/...`, it incorrectly treats `LLMBO` as the "netlists root",
  so it *cannot* see `netlists/_*.json`, and then falls back to local detection.
  That yields very small `perf_dim_max/sub_dim_max` and therefore a tiny feature dim.

This script fixes that by:
- Accepting `--netlists-root` explicitly (default: `netlists`) and using that
  for global meaning files and global maxima computations.

It intentionally does NOT modify `comb_graph_to_gnn.py`.

Usage
  python3 scripts/comb_graph_to_gnn_global.py \
    --in LLMBO/amp2_ati_new/ \
    --out-dir LLMBO/amp2_ati_new/ \
    --netlists-root netlists

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


NODE_TYPES = ["performance", "sub-structure", "parameter", "net", "device", "terminal"]
SUBCAT_SLOTS = 6


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def node_type_index(t: str) -> int:
    try:
        return NODE_TYPES.index(t)
    except ValueError:
        return -1


def extract_base_metric(name: str) -> str:
    suffixes = ["-trade-off", "-ambiguous", "-directly-proportional", "-inversely-proportional"]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def detect_performance_meanings(nodes: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for n in nodes:
        if n.get("type") == "performance":
            nid = n.get("id")
            if nid:
                base = extract_base_metric(nid)
                if base not in names:
                    names.append(base)

    preferred = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]
    ordered: List[str] = []
    lower_map = {n.lower(): n for n in names}
    for p in preferred:
        if p.lower() in lower_map:
            ordered.append(lower_map[p.lower()])
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def detect_substructure_types(nodes: List[Dict[str, Any]]) -> List[str]:
    types: List[str] = []
    for n in nodes:
        if n.get("type") == "substructure":
            nid = n.get("id")
            if nid not in types:
                types.append(nid)
    return types


def normalize_substructure_name(name: str) -> str:
    import re

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
    ]

    for patt, canon in rules:
        if re.search(patt, s):
            return canon

    s = re.sub(r"\b\d+\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else "unknown"


def build_feature_matrix(
    nodes: List[Dict[str, Any]],
    perf_meanings: List[str],
    substruct_types: List[str],
    perf_dim_max: int,
    sub_dim_max: int,
) -> Tuple[np.ndarray, int, int]:
    perf_dim = len(perf_meanings)
    sub_dim = len(substruct_types)
    meaning_dim = max(4, perf_dim_max + sub_dim_max)

    D = len(NODE_TYPES) + SUBCAT_SLOTS + meaning_dim
    N = len(nodes)
    features = np.zeros((N, D), dtype=np.float32)

    perf_map = {name: i for i, name in enumerate(perf_meanings)}
    sub_map = {name: i for i, name in enumerate(substruct_types)}

    for i, n in enumerate(nodes):
        nid = str(n.get("id", ""))
        ntype = str(n.get("type", ""))

        tidx = node_type_index(ntype)
        if tidx >= 0:
            features[i, tidx] = 1.0

        base = len(NODE_TYPES)
        if ntype == "performance":
            if nid.endswith("-ambiguous"):
                features[i, base + 1] = 1.0
            elif nid.endswith("-trade-off"):
                features[i, base + 2] = 1.0
            elif nid.endswith("-directly-proportional"):
                features[i, base + 3] = 1.0
            else:
                features[i, base + 0] = 1.0
        elif ntype == "parameter":
            if nid.endswith("-directly-proportional"):
                features[i, base + 1] = 1.0
            elif nid.endswith("-inversely-proportional"):
                features[i, base + 2] = 1.0
            else:
                features[i, base + 0] = 1.0
        elif ntype == "device":
            dtyp = str(n.get("device_type") or "").lower()
            if "pmos" in dtyp:
                features[i, base + 0] = 1.0
            elif "nmos" in dtyp:
                features[i, base + 1] = 1.0
            elif "pnp" in dtyp:
                features[i, base + 2] = 1.0
            elif "npn" in dtyp:
                features[i, base + 3] = 1.0
            elif "res" in dtyp or "resistor" in dtyp:
                features[i, base + 4] = 1.0
            elif "cap" in dtyp or "capacitor" in dtyp:
                features[i, base + 5] = 1.0
        elif ntype == "terminal":
            parts = nid.split(":")
            if len(parts) >= 3:
                role = parts[-1]
                if role == "D":
                    features[i, base + 0] = 1.0
                elif role == "G":
                    features[i, base + 1] = 1.0
                elif role == "S":
                    features[i, base + 2] = 1.0

        mbase = len(NODE_TYPES) + SUBCAT_SLOTS
        if ntype == "performance":
            base_metric = extract_base_metric(nid)
            idx = perf_map.get(base_metric)
            if idx is not None and idx < perf_dim_max:
                features[i, mbase + idx] = 1.0
        elif ntype in ("sub-structure", "substructure"):
            canon = normalize_substructure_name(nid)
            idx = sub_map.get(canon)
            if idx is not None and idx < sub_dim_max:
                features[i, mbase + perf_dim_max + idx] = 1.0

    print(
        f"Meaning dimension: {meaning_dim}, perf_dim={perf_dim}, sub_dim={sub_dim}, "
        f"perf_dim_max={perf_dim_max}, sub_dim_max={sub_dim_max}"
    )

    return features, D, meaning_dim


def build_adjacency(nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> np.ndarray:
    id2idx = {n["id"]: i for i, n in enumerate(nodes)}
    N = len(nodes)
    adj = np.zeros((N, N), dtype=np.uint8)
    for l in links:
        s = l.get("source")
        t = l.get("target")
        if s not in id2idx or t not in id2idx:
            continue
        si = id2idx[s]
        ti = id2idx[t]
        adj[si, ti] = 1
        adj[ti, si] = 1
    return adj


def load_global_meanings(netlists_root: str) -> Tuple[List[str], List[str]]:
    gp = load_json(os.path.join(netlists_root, "_performance_meanings.json"))
    gs = load_json(os.path.join(netlists_root, "_substructures_ordered.json"))

    perf_meanings = gp.get("performance_meanings") or []
    substruct_types = gs.get("ordered_substructures") or gs.get("ordered_substructure") or []

    return list(perf_meanings), list(substruct_types)


def compute_maxima_like_training(netlists_root: str) -> Tuple[int, int]:
    """Replicate `comb_graph_to_gnn.py`'s perf/sub maxima computation.

    The training feature width was produced by:
    - `perf_dim_max`: max over per-family performance meanings lists (often small)
    - `sub_dim_max`: max over per-family substructure ordered lists (can be large)

    If some family-level files are missing (common in this repo), we fall back to
    an empirically correct default by reading an existing `comb_graph_gnn_meta.json`
    under `netlists/`.
    """
    perf_dim_max = 0
    sub_dim_max = 0

    for fam in sorted(os.listdir(netlists_root)):
        fam_dir = os.path.join(netlists_root, fam)
        if not os.path.isdir(fam_dir):
            continue
        if fam.startswith("_"):
            continue

        pf: List[str] = []
        perff = os.path.join(fam_dir, "_performance_meanings.json")
        if os.path.exists(perff):
            try:
                pobj = load_json(perff)
                pf = pobj.get("performance_meanings") or []
            except Exception:
                pf = []

        ss: List[str] = []
        subsf = os.path.join(fam_dir, "_substructures_ordered.json")
        if os.path.exists(subsf):
            try:
                sobj = load_json(subsf)
                ss = sobj.get("ordered_substructures") or sobj.get("ordered_substructure") or []
            except Exception:
                ss = []

        perf_dim_max = max(perf_dim_max, len(pf))
        sub_dim_max = max(sub_dim_max, len(ss))

    # If family-level files are sparse, read an existing meta to get the true widths.
    if perf_dim_max == 0 or sub_dim_max == 0:
        for root, _dirs, files in os.walk(netlists_root):
            if "comb_graph_gnn_meta.json" in files:
                try:
                    meta = load_json(os.path.join(root, "comb_graph_gnn_meta.json"))
                    perf_dim_max = max(perf_dim_max, int(meta.get("perf_dim_max", 0)))
                    sub_dim_max = max(sub_dim_max, int(meta.get("sub_dim_max", 0)))
                    if perf_dim_max and sub_dim_max:
                        break
                except Exception:
                    continue

    perf_dim_max = max(perf_dim_max, 1)
    sub_dim_max = max(sub_dim_max, 1)
    return int(perf_dim_max), int(sub_dim_max)


def compute_global_maxima_from_training_meta(netlists_root: str) -> Tuple[int, int]:
    """Compute widths from an existing netlists meta file.

    This ensures we match the exact training feature width even if some family
    folders are missing auxiliary files.
    """
    for root, _dirs, files in os.walk(netlists_root):
        if "comb_graph_gnn_meta.json" in files:
            meta = load_json(os.path.join(root, "comb_graph_gnn_meta.json"))
            pdm = int(meta.get("perf_dim_max", 0) or 0)
            sdm = int(meta.get("sub_dim_max", 0) or 0)
            if pdm > 0 and sdm > 0:
                return pdm, sdm
    # final fallback
    return 6, 311


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert comb_graph.json into fixed-schema GNN arrays")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=False)
    ap.add_argument(
        "--netlists-root",
        default="netlists",
        help="Path to the training netlists root (contains _performance_meanings.json, etc.).",
    )
    ap.add_argument(
        "--perf-dim-max",
        type=int,
        default=None,
        help="Override perf_dim_max (defaults to len(netlists/_performance_meanings.json)).",
    )
    ap.add_argument(
        "--sub-dim-max",
        type=int,
        default=None,
        help="Override sub_dim_max (defaults to the training-time max used under netlists).",
    )
    ap.add_argument(
        "--maxima-source",
        choices=["like-training", "training-meta"],
        default="like-training",
        help="How to choose perf_dim_max/sub_dim_max.",
    )
    args = ap.parse_args()

    in_path = args.in_path
    if os.path.isdir(in_path):
        # Match repository batch behavior: prefer comb_graph.json, else <id>_comb_graph.json
        d = in_path
        cg = os.path.join(d, "comb_graph.json")
        if os.path.exists(cg):
            in_path = cg
        else:
            cid = os.path.basename(os.path.normpath(d))
            alt = os.path.join(d, f"{cid}_comb_graph.json")
            if os.path.exists(alt):
                in_path = alt
            else:
                # Fallback: if the directory name isn't the circuit id, accept any *_comb_graph.json
                matches = sorted(
                    f
                    for f in (os.path.join(d, fn) for fn in os.listdir(d))
                    if os.path.isfile(f) and f.endswith("_comb_graph.json")
                )
                if len(matches) == 1:
                    in_path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"No comb graph JSON found in directory {d}. Expected {cg} or {alt}"
                    )
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Comb graph JSON not found at {in_path}")

    out_dir = args.out_dir or os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    netlists_root = args.netlists_root
    if not os.path.isdir(netlists_root):
        raise FileNotFoundError(f"netlists root not found: {netlists_root}")

    data = load_json(in_path)
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    # Always use the global meaning vocabularies
    perf_meanings, substruct_types = load_global_meanings(netlists_root)

    if args.maxima_source == "like-training":
        perf_dim_max, sub_dim_max = compute_maxima_like_training(netlists_root)
    else:
        perf_dim_max, sub_dim_max = compute_global_maxima_from_training_meta(netlists_root)
    if args.perf_dim_max is not None:
        perf_dim_max = int(args.perf_dim_max)
    if args.sub_dim_max is not None:
        sub_dim_max = int(args.sub_dim_max)

    features, D, meaning_dim = build_feature_matrix(
        nodes,
        perf_meanings=perf_meanings,
        substruct_types=substruct_types,
        perf_dim_max=perf_dim_max,
        sub_dim_max=sub_dim_max,
    )
    adj = build_adjacency(nodes, links)

    nodes_ids = [n["id"] for n in nodes]
    npz_path = os.path.join(out_dir, "comb_graph_gnn.npz")
    np.savez_compressed(
        npz_path,
        nodes=np.array(nodes_ids, dtype=object),
        features=features,
        adjacency=adj,
    )
    print(f"Wrote NPZ to {npz_path}")

    meta = {
        "feature_dim": int(D),
        "type_order": NODE_TYPES,
        "subcat_slots": SUBCAT_SLOTS,
        "meaning_dim": int(meaning_dim),
        "performance_dim": int(len(perf_meanings)),
        "substructure_dim": int(len(substruct_types)),
        "perf_dim_max": int(perf_dim_max),
        "sub_dim_max": int(sub_dim_max),
        "performance_meanings": perf_meanings,
        "substructure_types": substruct_types,
        "netlists_root": os.path.abspath(netlists_root),
    }

    meta_path = os.path.join(out_dir, "comb_graph_gnn_meta.json")
    write_json(meta, meta_path)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
