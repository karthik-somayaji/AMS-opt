import json
import re
import argparse
from typing import Dict, Any

# ------------------------------------------------------------------
# Core parser
# ------------------------------------------------------------------

MOS_LIKE_MODELS = {
    "nmos", "pmos", "nmos4", "pmos4", "mos", "mos3", "mos4"
}

def add_node(graph: Dict[str, Any], node_id: str, node_type: str, **attrs):
    """Add a node if it does not already exist."""
    if "nodes_index" not in graph:
        graph["nodes_index"] = {}
    if node_id in graph["nodes_index"]:
        return

    node = {"id": node_id, "type": node_type}
    node.update(attrs)
    graph["nodes"].append(node)
    graph["nodes_index"][node_id] = node


def add_link(graph: Dict[str, Any], source: str, target: str):
    """Add an edge (link) to the graph."""
    graph["links"].append({"source": source, "target": target})


def parse_device_line(line: str):
    """
    Parse a single device line of the form:
        M2 (VOUT1 net14 VDD VDD) pmos4
        R1 (VOUT1 net14) resistor
    Returns (dev_name, node_list, dev_type) or None if not a device.
    """
    # Remove inline comments after '*', '//', or ';'
    line = re.split(r'[\*;]', line, maxsplit=1)[0]
    line = re.split(r'//', line, maxsplit=1)[0].strip()
    if not line:
        return None

    # Match: <name> (<nodes...>) <type> ...
    m = re.match(r'^(\S+)\s*\(([^)]*)\)\s*([^\s]+)?', line)
    if not m:
        return None

    dev_name = m.group(1)
    nodes_str = m.group(2).strip()
    dev_type = (m.group(3) or "").strip()
    node_list = nodes_str.split()
    return dev_name, node_list, dev_type


def handle_mos_device(graph: Dict[str, Any],
                      dev_name: str,
                      node_list,
                      dev_type: str):
    """
    Build nodes/edges for a MOS device as 4 graph nodes:
        dev:Mx  (type='device')
        term:Mx:D (type='terminal')
        term:Mx:G (type='terminal')
        term:Mx:S (type='terminal')

    We assume a 3-terminal MOS abstraction:
        D = node_list[0]
        G = node_list[1]
        S = node_list[2]
    If 4 nodes are present and node_list[2] == node_list[3], we treat this
    as a 3-terminal device with source and bulk shorted.
    """
    # Ensure we have at least D,G,S; if fewer, bail out
    if len(node_list) < 3:
        raise ValueError(f"MOS device {dev_name} has < 3 nodes: {node_list}")

    drain_net = node_list[0]
    gate_net = node_list[1]
    source_net = node_list[2]

    # Create main MOS device node
    dev_id = f"dev:{dev_name}"
    add_node(graph, dev_id, "device", device_type=dev_type)

    # Create terminal nodes
    term_ids = {
        "D": f"term:{dev_name}:D",
        "G": f"term:{dev_name}:G",
        "S": f"term:{dev_name}:S",
    }

    for role, term_id in term_ids.items():
        add_node(graph, term_id, "terminal", device=dev_name, role=role)
        add_link(graph, dev_id, term_id)  # connect device to its terminal

    # Connect terminals to their nets
    term_to_net = {
        term_ids["D"]: drain_net,
        term_ids["G"]: gate_net,
        term_ids["S"]: source_net,
    }

    for term_id, net_name in term_to_net.items():
        net_id = f"net:{net_name}"
        add_node(graph, net_id, "net")
        add_link(graph, term_id, net_id)


def handle_generic_device(graph: Dict[str, Any],
                          dev_name: str,
                          node_list,
                          dev_type: str):
    """
    For non-MOS devices, we create:
        dev:<name> (type='device')
        net:<node> (type='net')
    with edges dev:<name> -> net:<node> for all connected nets.
    """
    dev_id = f"dev:{dev_name}"
    add_node(graph, dev_id, "device", device_type=dev_type)

    for n in node_list:
        net_id = f"net:{n}"
        add_node(graph, net_id, "net")
        add_link(graph, dev_id, net_id)


def netlist_to_graph_json(netlist_text: str) -> Dict[str, Any]:
    """
    Main entry: parse multiline netlist text into a JSON-serializable graph
    with nodes and links.
    """
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "links": [],
    }

    for raw_line in netlist_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('*'):
            continue

        parsed = parse_device_line(line)
        if parsed is None:
            continue

        dev_name, node_list, dev_type = parsed
        dev_type_lower = dev_type.lower()

        if dev_type_lower in MOS_LIKE_MODELS or dev_name[0].upper() == "M":
            # Treat as MOS-like device (3-terminal abstraction)
            handle_mos_device(graph, dev_name, node_list, dev_type)
        else:
            # resistor, source, etc.
            handle_generic_device(graph, dev_name, node_list, dev_type)

    # Internal index is not part of output JSON
    graph.pop("nodes_index", None)
    return graph


def write_graph_json_from_netlist(netlist_text: str, outfile: str):
    graph = netlist_to_graph_json(netlist_text)
    with open(outfile, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Wrote graph JSON to {outfile}")


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert netlist to circuit graph JSON"
    )
    parser.add_argument(
        "--netlist-path",
        type=str,
        required=True,
        help="Path to the input netlist file"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file"
    )
    args = parser.parse_args()

    # Read netlist from file
    with open(args.netlist_path, "r") as f:
        netlist = f.read()

    write_graph_json_from_netlist(netlist, args.output_jsonl)
