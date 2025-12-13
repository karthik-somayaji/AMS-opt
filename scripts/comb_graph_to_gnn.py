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

# Sub-category slots (4)
# performance -> [original, ambiguous, trade-off, directly-proportional]
# parameter -> [original, directly-proportional, inversely-proportional, unused]
# device -> [pmos4, nmos4, unused, unused]
# terminal -> [D, G, S, unused]
SUBCAT_SLOTS = 4


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


def detect_performance_meanings(nodes: List[Dict[str, Any]]) -> List[str]:
    names = []
    for n in nodes:
        if n.get("type") == "performance":
            nid = n.get("id")
            if nid not in names:
                names.append(nid)
    # ensure common ones first
    order = ["Gain", "CMRR", "UGF", "Power"]
    for o in reversed(order):
        if o in names:
            names.remove(o)
            names.insert(0, o)
    return names


def detect_substructure_types(nodes: List[Dict[str, Any]]) -> List[str]:
    types = []
    for n in nodes:
        if n.get("type") == "sub-structure":
            nid = n.get("id")
            if nid not in types:
                types.append(nid)
    return types


def build_feature_matrix(nodes: List[Dict[str, Any]], perf_meanings: List[str], substruct_types: List[str]):
    meaning_dim = max(4, len(substruct_types), len(perf_meanings))
    D = len(NODE_TYPES) + SUBCAT_SLOTS + meaning_dim
    N = len(nodes)
    features = [[0.0] * D for _ in range(N)]

    perf_map = {name: i for i, name in enumerate(perf_meanings)}
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

        # meaning vector (last meaning_dim slots)
        mbase = len(NODE_TYPES) + SUBCAT_SLOTS
        if ntype == "performance":
            # map by exact id
            idx = perf_map.get(nid)
            if idx is not None and idx < meaning_dim:
                features[i][mbase + idx] = 1.0
        elif ntype == "sub-structure":
            idx = sub_map.get(nid)
            if idx is not None and idx < meaning_dim:
                features[i][mbase + idx] = 1.0

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

    perf_meanings = detect_performance_meanings(nodes)
    substruct_types = detect_substructure_types(nodes)

    features_list, D, meaning_dim = build_feature_matrix(nodes, perf_meanings, substruct_types)
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
        "performance_meanings": perf_meanings,
        "substructure_types": substruct_types,
    }
    meta_path = os.path.join(out_dir, "comb_graph_gnn_meta.json")
    write_json(meta, meta_path)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
