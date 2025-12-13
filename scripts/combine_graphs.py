#!/usr/bin/env python3
"""
Combine a structure graph (`str_graph.json`) and a functional/parameter graph (`fun_graph.json`) into
a single combined graph. Also adds links from MOS device nodes in the structure graph to the
corresponding parameter nodes (e.g. `dev:M2` -> `W_M2`, `L_M2`).

Usage:
  python scripts/combine_graphs.py --str_graph path/to/str_graph.json \
      --fun_graph path/to/fun_graph.json --out path/to/comb_graph.json
"""
import argparse
import json
import os
from typing import Dict, List, Any, Tuple


MOS_KEYWORDS = ("mos", "nmos", "pmos")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def node_list_to_dict(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    d: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        d[n["id"]] = dict(n)
    return d


def merge_nodes(str_nodes: List[Dict[str, Any]], fun_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Start from fun_nodes (parameters, high-level) then overlay/extend with str_nodes
    merged = node_list_to_dict(fun_nodes or [])
    for n in (str_nodes or []):
        node_id = n["id"]
        if node_id in merged:
            # merge attributes (structure nodes take precedence)
            merged[node_id].update(n)
        else:
            merged[node_id] = dict(n)
    return list(merged.values())


def link_key(l: Dict[str, Any]) -> Tuple[str, str, Tuple[Tuple[str, Any], ...]]:
    extras = tuple(sorted((k, v) for k, v in l.items() if k not in ("source", "target")))
    return (l.get("source"), l.get("target"), extras)


def merge_links(str_links: List[Dict[str, Any]], fun_links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for l in (str_links or []) + (fun_links or []):
        k = link_key(l)
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(l))
    return out


def is_mos_like(node: Dict[str, Any]) -> bool:
    if node.get("type") != "device":
        return False
    dev_type = (node.get("device_type") or "").lower()
    if any(k in dev_type for k in MOS_KEYWORDS):
        return True
    # fallback: device id like dev:M0, dev:M1 etc.
    node_id = node.get("id", "")
    if node_id.lower().startswith("dev:m"):
        return True
    return False


def add_device_parameter_links(combined_nodes: List[Dict[str, Any]],
                               combined_links: List[Dict[str, Any]],
                               str_nodes: List[Dict[str, Any]]) -> None:
    nodes_by_id = {n["id"]: n for n in combined_nodes}
    existing_link_keys = {link_key(l) for l in combined_links}

    for dev in (str_nodes or []):
        if not is_mos_like(dev):
            continue
        dev_id = dev["id"]  # e.g. 'dev:M2'
        # extract short name after 'dev:' if present
        short = dev_id.split(":", 1)[1] if ":" in dev_id else dev_id
        # parameter node names expected in fun_graph: W_<short>, L_<short>
        for pname in (f"W_{short}", f"L_{short}"):
            if pname not in nodes_by_id:
                # create a parameter node if not present
                param_node = {"id": pname, "type": "parameter"}
                combined_nodes.append(param_node)
                nodes_by_id[pname] = param_node
            new_link = {"source": dev_id, "target": pname}
            if link_key(new_link) not in existing_link_keys:
                combined_links.append(new_link)
                existing_link_keys.add(link_key(new_link))


def build_combined_graph(str_graph: Dict[str, Any], fun_graph: Dict[str, Any]) -> Dict[str, Any]:
    str_nodes = str_graph.get("nodes", [])
    fun_nodes = fun_graph.get("nodes", [])
    str_links = str_graph.get("links", [])
    fun_links = fun_graph.get("links", [])

    combined_nodes = merge_nodes(str_nodes, fun_nodes)
    combined_links = merge_links(str_links, fun_links)

    add_device_parameter_links(combined_nodes, combined_links, str_nodes)

    # preserve top-level metadata from str_graph when available, else make minimal
    out: Dict[str, Any] = {}
    # copy non-nodes/links keys from str_graph (e.g., directed, multigraph, graph)
    for k, v in str_graph.items():
        if k not in ("nodes", "links"):
            out[k] = v

    out["nodes"] = combined_nodes
    out["links"] = combined_links
    return out


def main():
    p = argparse.ArgumentParser(description="Combine structure and functional graphs into one graph")
    p.add_argument("--str_graph", required=True, help="Path to structure graph JSON (str_graph.json)")
    p.add_argument("--fun_graph", required=True, help="Path to functional graph JSON (fun_graph.json)")
    p.add_argument("--out", required=True, help="Path to write combined graph JSON")
    args = p.parse_args()

    # If the provided paths are directories, look for the expected filenames inside them.
    def resolve_input(path: str, default_filename: str) -> str:
        if os.path.isdir(path):
            candidate = os.path.join(path, default_filename)
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Expected file '{default_filename}' inside directory {path}")
            return candidate
        return path

    str_path = resolve_input(args.str_graph, "str_graph.json")
    fun_path = resolve_input(args.fun_graph, "fun_graph.json")

    # Resolve output: if directory provided, create comb_graph.json inside it
    out_path = args.out
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "comb_graph.json")

    str_graph = load_json(str_path)
    fun_graph = load_json(fun_path)

    combined = build_combined_graph(str_graph, fun_graph)
    write_json(combined, out_path)
    print(f"Wrote combined graph to {out_path}")


if __name__ == "__main__":
    main()
