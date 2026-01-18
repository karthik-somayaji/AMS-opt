#!/usr/bin/env python3
"""
Convert a combined graph JSON (`comb_graph.json`) into GNN-ready arrays:
- `nodes`: list of node ids
- `features`: NxD feature matrix (D described in output metadata)
- `adjacency`: NxN adjacency matrix (0/1)

Feature layout (flexible):
- first 6 dims: one-hot node type [performance, sub-structure, parameter, net, device, terminal]
- next 4 dims: sub-category (meanings depend on node type)
- last M dims: meaning encoding — at least 4 for performance meanings, extended if there are more sub-structure types

The script writes a `.npz` file with arrays `nodes`, `features`, `adjacency` and a `.json` metadata file explaining mappings.

Usage:
  python scripts/comb_graph_to_gnn.py --in path/to/comb_graph.json --out-dir path/to/output_dir
If `--in` is a directory, looks for `comb_graph.json` inside it.
"""
import argparse
import json
import os
from typing import Dict, List, Any

try:
    import numpy as np
except Exception:
    np = None


NODE_TYPES = ["performance", "sub-structure", "parameter", "net", "device", "terminal"]

# Sub-category slots (6)
# performance -> [original, ambiguous, trade-off, directly-proportional, unused, unused]
# parameter -> [original, directly-proportional, inversely-proportional, unused, unused, unused]
# device -> [pmos4, nmos4, pnp, npn, resistor, capacitor]
# terminal -> [D, G, S, unused, unused, unused]
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
    """Extract base metric name by removing variant suffixes."""
    suffixes = ["-trade-off", "-ambiguous", "-directly-proportional", "-inversely-proportional"]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def detect_performance_meanings(nodes: List[Dict[str, Any]]) -> List[str]:
    """Detect base performance metrics (without variants) from nodes."""
    names = []
    for n in nodes:
        if n.get("type") == "performance":
            nid = n.get("id")
            if nid:
                base = extract_base_metric(nid)
                if base not in names:
                    names.append(base)
    # preferred ordering for known metrics
    preferred = ["Gain", "CMRR", "UGF", "Power", "Delay", "Offset", "Hysteresis"]
    ordered = []
    lower_map = {n.lower(): n for n in names}
    for p in preferred:
        if p.lower() in lower_map:
            ordered.append(lower_map[p.lower()])
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def detect_substructure_types(nodes: List[Dict[str, Any]]) -> List[str]:
    types = []
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

    # Local mapping rules (keep in sync with scripts/collect_substructures.py)
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
    if s == "":
        return "unknown"
    return s


def detect_substructure_presence(nodes: List[Dict[str, Any]], ordered_subs: List[str]) -> List[str]:
    """Return ordered list of substructure types (ordered_subs) and ensure presence mapping.

    This function checks which canonical substructures from `ordered_subs` are present in `nodes`.
    """
    present = set()
    for n in nodes:
        if n.get("type") == "substructure":
            raw = n.get("id") or n.get("name") or ""
            canon = normalize_substructure_name(raw)
            if canon in ordered_subs:
                present.add(canon)

    # We return the ordered_subs so upstream can build a one-hot using this ordering.
    return ordered_subs


def build_feature_matrix(nodes: List[Dict[str, Any]], perf_meanings: List[str], substruct_types: List[str], perf_dim_max: int, sub_dim_max: int):
    perf_dim = len(perf_meanings)
    sub_dim = len(substruct_types)
    meaning_dim = max(4, perf_dim_max + sub_dim_max)
    print(f"Meaning dimension: {meaning_dim}, perf_dim={perf_dim}, sub_dim={sub_dim}, perf_dim_max={perf_dim_max}, sub_dim_max={sub_dim_max}")
    D = len(NODE_TYPES) + SUBCAT_SLOTS + meaning_dim
    N = len(nodes)
    features = [[0.0] * D for _ in range(N)]

    perf_map = {name: i for i, name in enumerate(perf_meanings)}
    # substruct_types may contain canonical names; build mapping by canonical name
    sub_map = {name: i for i, name in enumerate(substruct_types)}

    for i, n in enumerate(nodes):
        nid = n.get("id")
        ntype = n.get("type")
        # type one-hot
        tidx = node_type_index(ntype)
        if tidx >= 0:
            features[i][tidx] = 1.0

        # sub-category slots (next 4)
        base = len(NODE_TYPES)
        if ntype == "performance":
            # original vs variants
            if nid.endswith("-ambiguous"):
                features[i][base + 1] = 1.0
            elif nid.endswith("-trade-off"):
                features[i][base + 2] = 1.0
            elif nid.endswith("-directly-proportional"):
                features[i][base + 3] = 1.0
            else:
                features[i][base + 0] = 1.0
        elif ntype == "parameter":
            if nid.endswith("-directly-proportional"):
                features[i][base + 1] = 1.0
            elif nid.endswith("-inversely-proportional"):
                features[i][base + 2] = 1.0
            else:
                features[i][base + 0] = 1.0
        elif ntype == "device":
            dtyp = (n.get("device_type") or "").lower()
            if "pmos" in dtyp:
                features[i][base + 0] = 1.0
            elif "nmos" in dtyp:
                features[i][base + 1] = 1.0
            elif "pnp" in dtyp:
                features[i][base + 2] = 1.0
            elif "npn" in dtyp:
                features[i][base + 3] = 1.0
            elif "res" in dtyp or "resistor" in dtyp:
                features[i][base + 4] = 1.0
            elif "cap" in dtyp or "capacitor" in dtyp:
                features[i][base + 5] = 1.0
        elif ntype == "terminal":
            # terminal id like 'term:M1:D'
            parts = nid.split(":")
            if len(parts) >= 3:
                role = parts[-1]
                if role == "D":
                    features[i][base + 0] = 1.0
                elif role == "G":
                    features[i][base + 1] = 1.0
                elif role == "S":
                    features[i][base + 2] = 1.0

        # meaning vector (last meaning_dim slots). layout: [perf_block (perf_dim_max) | sub_block (sub_dim_max)]
        mbase = len(NODE_TYPES) + SUBCAT_SLOTS
        if ntype == "performance":
            # map by base metric (not the full id with variant suffix) into the perf block (left side)
            base_metric = extract_base_metric(nid)
            idx = perf_map.get(base_metric)
            if idx is not None and idx < perf_dim_max:
                features[i][mbase + idx] = 1.0
        elif ntype in ("sub-structure", "substructure"):
            # normalize raw id to canonical and map to ordered list; place into sub block after perf_dim_max
            canon = normalize_substructure_name(nid)
            idx = sub_map.get(canon)
            if idx is not None and idx < sub_dim_max:
                features[i][mbase + perf_dim_max + idx] = 1.0

    return features, D, meaning_dim


def build_adjacency(nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]):
    id2idx = {n["id"]: i for i, n in enumerate(nodes)}
    N = len(nodes)
    adj = [[0] * N for _ in range(N)]
    for l in links:
        s = l.get("source")
        t = l.get("target")
        if s not in id2idx or t not in id2idx:
            continue
        si = id2idx[s]
        ti = id2idx[t]
        adj[si][ti] = 1
        adj[ti][si] = 1
    return adj


def main():
    p = argparse.ArgumentParser(description="Convert comb_graph.json into GNN-ready arrays")
    p.add_argument("--in", dest="in_path", required=True, help="Path to comb_graph.json or directory containing it")
    p.add_argument("--out-dir", dest="out_dir", required=False, help="Directory to write outputs (defaults to input dir)")
    args = p.parse_args()

    in_path = args.in_path
    if os.path.isdir(in_path):
        in_path = os.path.join(in_path, "comb_graph.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"comb_graph.json not found at {in_path}")

    out_dir = args.out_dir or os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    data = load_json(in_path)
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    # determine family and netlists root
    in_dir = os.path.dirname(in_path)
    family_dir = os.path.dirname(in_dir)
    netlists_root = os.path.dirname(family_dir)

    # Prefer global performance meanings/substructures (netlists/_*.json). Fall back to family-level files or detect locally.
    perf_meanings = []
    substruct_types = None
    global_perf_path = os.path.join(netlists_root, "_performance_meanings.json")
    global_subs_path = os.path.join(netlists_root, "_substructures_ordered.json")

    if os.path.exists(global_perf_path):
        try:
            gp = load_json(global_perf_path)
            perf_meanings = gp.get("performance_meanings") or []
        except Exception:
            perf_meanings = []
    else:
        perf_path = os.path.join(family_dir, "_performance_meanings.json")
        if os.path.exists(perf_path):
            try:
                pobj = load_json(perf_path)
                perf_meanings = pobj.get("performance_meanings") or []
            except Exception:
                perf_meanings = []
    if not perf_meanings:
        perf_meanings = detect_performance_meanings(nodes)

    # Determine family directory and try to load ordered substructures produced by collector
    # substructures: prefer global then family
    if os.path.exists(global_subs_path):
        try:
            gs = load_json(global_subs_path)
            substruct_types = gs.get("ordered_substructures") or gs.get("ordered_substructure") or []
        except Exception:
            substruct_types = None
    else:
        ordered_subs_path = os.path.join(family_dir, "_substructures_ordered.json")
        substruct_types = None
        if os.path.exists(ordered_subs_path):
            try:
                subs_obj = load_json(ordered_subs_path)
                substruct_types = subs_obj.get("ordered_substructures") or subs_obj.get("ordered_substructure") or []
            except Exception:
                substruct_types = None

    if not substruct_types:
        # fallback: detect from this circuit alone
        substruct_types = detect_substructure_types(nodes)
    else:
        # If an ordered list exists, ensure we detect presence (will be encoded by index)
        _ = detect_substructure_presence(nodes, substruct_types)

    # Compute per-family maxima across all families under netlists_root to ensure fixed feature widths
    perf_dim_max = 0
    sub_dim_max = 0
    try:
        for fam in sorted(os.listdir(netlists_root)):
            fam_dir = os.path.join(netlists_root, fam)
            if not os.path.isdir(fam_dir):
                continue
            # load perf list
            pf = []
            perff = os.path.join(fam_dir, "_performance_meanings.json")
            if os.path.exists(perff):
                try:
                    pobj = load_json(perff)
                    pf = pobj.get("performance_meanings") or []
                except Exception:
                    pf = []
            else:
                # attempt to detect from comb_graph.json files in the family
                pf_nodes = []
                for cd in sorted(os.listdir(fam_dir)):
                    cdpath = os.path.join(fam_dir, cd)
                    combp = os.path.join(cdpath, "comb_graph.json")
                    if os.path.exists(combp):
                        try:
                            cdata = load_json(combp)
                            for n in cdata.get("nodes", []):
                                if n.get("type") == "performance":
                                    nid = n.get("id")
                                    if nid:
                                        # Extract base metric only (without variant suffix)
                                        base = extract_base_metric(nid)
                                        if base not in pf_nodes:
                                            pf_nodes.append(base)
                        except Exception:
                            continue
                pf = detect_performance_meanings([{"type":"performance","id":n} for n in pf_nodes]) if pf_nodes else []

            ss = []
            subsf = os.path.join(fam_dir, "_substructures_ordered.json")
            if os.path.exists(subsf):
                try:
                    sobj = load_json(subsf)
                    ss = sobj.get("ordered_substructures") or []
                except Exception:
                    ss = []
            else:
                # detect from comb_graph.json files
                ss_nodes = []
                for cd in sorted(os.listdir(fam_dir)):
                    cdpath = os.path.join(fam_dir, cd)
                    combp = os.path.join(cdpath, "comb_graph.json")
                    if os.path.exists(combp):
                        try:
                            cdata = load_json(combp)
                            for n in cdata.get("nodes", []):
                                if n.get("type") == "substructure":
                                    nid = n.get("id")
                                    if nid and nid not in ss_nodes:
                                        ss_nodes.append(nid)
                        except Exception:
                            continue
                ss = ss_nodes

            perf_dim_max = max(perf_dim_max, len(pf))
            sub_dim_max = max(sub_dim_max, len(ss))
    except Exception:
        # If anything fails, fallback to local sizes
        perf_dim_max = max(perf_dim_max, len(perf_meanings))
        sub_dim_max = max(sub_dim_max, len(substruct_types))
    # Ensure minimum slots
    perf_dim_max = max(perf_dim_max, 1)
    sub_dim_max = max(sub_dim_max, 1)
    

    features_list, D, meaning_dim = build_feature_matrix(nodes, perf_meanings, substruct_types, perf_dim_max, sub_dim_max)
    adj = build_adjacency(nodes, links)

    # Convert to numpy if available
    nodes_ids = [n["id"] for n in nodes]
    if np is not None:
        features = np.asarray(features_list, dtype=np.float32)
        adjacency = np.asarray(adj, dtype=np.uint8)
        npz_path = os.path.join(out_dir, "comb_graph_gnn.npz")
        np.savez_compressed(npz_path, nodes=np.array(nodes_ids, dtype=object), features=features, adjacency=adjacency)
        print(f"Wrote NPZ to {npz_path}")
    else:
        # fallback to JSON
        json_out = {"nodes": nodes_ids, "features": features_list, "adjacency": adj}
        json_path = os.path.join(out_dir, "comb_graph_gnn.json")
        write_json(json_out, json_path)
        print(f"Wrote JSON to {json_path}")

    # Write metadata
    meta = {
        "feature_dim": D,
        "type_order": NODE_TYPES,
        "subcat_slots": SUBCAT_SLOTS,
        "meaning_dim": meaning_dim,
        "performance_dim": len(perf_meanings),
        "substructure_dim": len(substruct_types),
        "perf_dim_max": perf_dim_max,
        "sub_dim_max": sub_dim_max,
        "performance_meanings": perf_meanings,
        "substructure_types": substruct_types,
    }
    meta_path = os.path.join(out_dir, "comb_graph_gnn_meta.json")
    write_json(meta, meta_path)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
