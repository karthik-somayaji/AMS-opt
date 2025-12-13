#!/usr/bin/env python3
"""
Transform a `fun_graph.json` by expanding performance and parameter nodes into
variant nodes and collapsing relation-typed edges into simple "connects" links
between the appropriate variant nodes.

Example:
  {"source": "W_M0", "target": "Gain", "relation": "directly-proportional"}
becomes
  {"source": "W_M0-directly-proportional", "target": "Gain-directly-proportional", "relation": "connects"}

Usage:
  python scripts/transform_fun_graph.py --in path/to/fun_graph.json --out path/to/fun_updated.json
If `--out` is a directory, writes `fun_updated.json` inside it.
"""
import argparse
import json
import os
from typing import Dict, List, Any, Set, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_node_index(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in nodes}


def create_variant_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of additional nodes to add.

    - performance nodes: add `-ambiguous`, `-trade-off`, `-directly-proportional`
    - parameter nodes: add `-directly-proportional`, `-inversely-proportional`
    - substructure nodes: no variants
    """
    extras: List[Dict[str, Any]] = []
    for n in nodes:
        nid = n.get("id")
        ntype = n.get("type")
        if ntype == "performance":
            for suf in ("ambiguous", "trade-off", "directly-proportional"):
                extras.append({"id": f"{nid}-{suf}", "type": ntype})
        elif ntype == "parameter":
            for suf in ("directly-proportional", "inversely-proportional"):
                extras.append({"id": f"{nid}-{suf}", "type": ntype})
    return extras


def map_endpoint(node_id: str, node_index: Dict[str, Dict[str, Any]], relation: str, side: str) -> str:
    """Map a node id and relation to the proper (possibly variant) node id.

    side is either 'source' or 'target' — included for future extensibility.
    Rules (best-effort mapping based on provided instructions):
    - 'directly-proportional': map performance -> '-directly-proportional', parameter -> '-directly-proportional'
    - 'inversely-proportional': map parameter -> '-inversely-proportional', performance -> '-trade-off'
    - 'ambiguous': map performance -> '-ambiguous'
    - 'trade-off': map performance -> '-trade-off'
    - otherwise: return original node id
    """
    node = node_index.get(node_id)
    if not node:
        return node_id
    ntype = node.get("type")
    r = relation
    if r == "directly-proportional":
        if ntype in ("parameter", "performance"):
            return f"{node_id}-directly-proportional"
    if r == "inversely-proportional":
        if ntype == "parameter":
            return f"{node_id}-inversely-proportional"
        if ntype == "performance":
            return f"{node_id}-trade-off"
    if r == "ambiguous":
        if ntype == "performance":
            return f"{node_id}-ambiguous"
    if r == "trade-off":
        if ntype == "performance":
            return f"{node_id}-trade-off"
    # fallback: leave as-is for relations like 'belongs-to', 'influences', etc.
    return node_id


def transform_links(links: List[Dict[str, Any]], node_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for l in links:
        src = l.get("source")
        tgt = l.get("target")
        rel = l.get("relation")
        new_src = map_endpoint(src, node_index, rel, "source")
        new_tgt = map_endpoint(tgt, node_index, rel, "target")
        key = (new_src, new_tgt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": new_src, "target": new_tgt, "relation": "connects"})
    return out


def main():
    p = argparse.ArgumentParser(description="Transform fun_graph.json into collapsed connects graph with variant nodes")
    p.add_argument("--in", dest="in_path", required=True, help="Path to fun_graph.json")
    p.add_argument("--out", dest="out_path", required=False, help="Output path (file or directory)")
    args = p.parse_args()

    in_path = args.in_path
    out_path = args.out_path or None
    if os.path.isdir(in_path):
        in_path = os.path.join(in_path, "fun_graph.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input fun_graph not found: {in_path}")
    if out_path is None:
        out_path = os.path.join(os.path.dirname(in_path), "fun_updated.json")
    elif os.path.isdir(out_path):
        out_path = os.path.join(out_path, "fun_updated.json")

    data = load_json(in_path)
    orig_nodes = data.get("nodes", [])
    orig_links = data.get("links", [])

    extras = create_variant_nodes(orig_nodes)

    # Build index including originals so mapping can find node types
    node_index = build_node_index(orig_nodes)

    # Append extras to nodes list
    new_nodes = list(orig_nodes) + extras
    # Extend index with extras (they have same type as originals)
    for e in extras:
        node_index[e["id"]] = e

    new_links = transform_links(orig_links, build_node_index(orig_nodes))

    # Add variant connectivity: original performance node -> its variants,
    # and original parameter node -> its variants (directly/inversely proportional).
    seen = {(l["source"], l["target"]) for l in new_links}
    for n in orig_nodes:
        nid = n.get("id")
        ntype = n.get("type")
        if ntype == "performance":
            for suf in ("ambiguous", "trade-off", "directly-proportional"):
                tgt = f"{nid}-{suf}"
                if (nid, tgt) not in seen:
                    new_links.append({"source": nid, "target": tgt, "relation": "connects"})
                    seen.add((nid, tgt))
        elif ntype == "parameter":
            for suf in ("directly-proportional", "inversely-proportional"):
                tgt = f"{nid}-{suf}"
                if (nid, tgt) not in seen:
                    new_links.append({"source": nid, "target": tgt, "relation": "connects"})
                    seen.add((nid, tgt))

    out = {"nodes": new_nodes, "links": new_links}
    write_json(out, out_path)
    print(f"Wrote transformed fun graph to {out_path}")


if __name__ == "__main__":
    main()
