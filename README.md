# AMS-opt: Analog Circuit Graph Processing Pipeline

This repository contains a pipeline to convert analog circuit netlists into Graph Neural Network (GNN)-ready representations. The pipeline processes AnalogGenie differential amplifier circuits and transforms them through multiple stages of graph construction and feature engineering.

## Overview

The pipeline consists of four main stages:

1. **Netlist to Structural Graph** (`get_netlist_to_SG.py`)
   - Parse SPICE netlists into structured graphs with nodes (devices, nets, terminals) and links (connections)

2. **Transform Functional Graph** (`scripts/transform_fun_graph.py`)
   - Expand functional/parameter graphs with variant nodes for relation types
   - Collapse relation-typed edges into simple "connects" edges

3. **Combine Graphs** (`scripts/combine_graphs.py`)
   - Merge structural and functional graphs into a unified representation
   - Add device-to-parameter links for MOS transistors

4. **Build GNN Features** (`scripts/comb_graph_to_gnn.py`)
   - Convert combined graphs into GNN-ready feature matrices and adjacency matrices
   - Save as compressed NumPy format (.npz) for efficient loading

---

## Stage 1: Netlist to Structural Graph

### Purpose
Parse a SPICE netlist and build a directed graph representation where:
- Nodes represent devices, nets (wires), and terminals
- Edges represent physical connections in the circuit

### File
`get_netlist_to_SG.py`

### Inputs
- SPICE netlist file (`.cir` or `.sp`)

### Outputs
- `str_graph.json`: Structural graph with nodes and links

### Graph Structure

**Nodes:**
- `device` nodes: e.g., `dev:M0`, `dev:R1` (with `device_type` attribute: `nmos4`, `pmos4`, `resistor`, etc.)
- `net` nodes: e.g., `net:VDD`, `net:VSS` (wires/nets)
- `terminal` nodes: e.g., `term:M0:D`, `term:M0:G`, `term:M0:S` (device terminals: Drain, Gate, Source)

**Links:**
- Device → Terminal: e.g., `dev:M0` → `term:M0:D`
- Terminal → Net: e.g., `term:M0:D` → `net:VOUT`

### Example Usage

```bash
python get_netlist_to_SG.py \
  --netlist-path netlists/diff_amps/75/75.cir \
  --output-jsonl netlists/diff_amps/75/str_graph.json
```

### Key Functions
- `parse_device_line()`: Parse a single SPICE device line (e.g., "M0 (VOUT1 VIN1 IB1 VSS) nmos4")
- `netlist_to_graph_json()`: Convert entire netlist to graph structure
- `handle_mos_device()`: Special handling for MOS devices with 3-terminal abstraction

---

## Stage 2: Transform Functional Graph

### Purpose
Transform a functional/parameter graph to:
- Expand performance nodes into variant nodes (e.g., `Gain` → `Gain-ambiguous`, `Gain-trade-off`, `Gain-directly-proportional`)
- Expand parameter nodes into variant nodes (e.g., `W_M0` → `W_M0-directly-proportional`, `W_M0-inversely-proportional`)
- Replace relation-typed edges with simple "connects" edges between appropriate variants

### File
`scripts/transform_fun_graph.py`

### Inputs
- `fun_graph.json`: Original functional graph with relation types

### Outputs
- `fun_updated.json`: Transformed graph with variant nodes and simplified edges

### Graph Transformations

**Original Links:**
```json
{"source": "W_M0", "target": "Gain", "relation": "directly-proportional"}
```

**Transformed:**
```json
{"source": "W_M0-directly-proportional", "target": "Gain-directly-proportional", "relation": "connects"}
```

Also adds "variant connectivity" edges:
```json
{"source": "W_M0", "target": "W_M0-directly-proportional", "relation": "connects"}
{"source": "Gain", "target": "Gain-directly-proportional", "relation": "connects"}
```

### Variant Node Rules

- **Performance nodes** (e.g., Gain, CMRR, UGF, Power):
  - Original: `Gain`
  - Variants: `Gain-ambiguous`, `Gain-trade-off`, `Gain-directly-proportional`

- **Parameter nodes** (e.g., W_M0, L_M1):
  - Original: `W_M0`
  - Variants: `W_M0-directly-proportional`, `W_M0-inversely-proportional`

- **Substructure nodes**: No variants (e.g., "M0-M1 differential pair" stays as-is)

### Example Usage

```bash
python scripts/transform_fun_graph.py \
  --in netlists/diff_amps/75/fun_graph.json \
  --out netlists/diff_amps/75/fun_updated.json
```

Or using directory shortcut:
```bash
python scripts/transform_fun_graph.py \
  --in netlists/diff_amps/75/ \
  --out netlists/diff_amps/75/
```

### Key Functions
- `create_variant_nodes()`: Generate additional variant nodes
- `map_endpoint()`: Map node + relation type to appropriate variant node
- `transform_links()`: Rewrite links to use variant nodes and dedup

---

## Stage 3: Combine Graphs

### Purpose
Merge structural and functional graphs into a unified representation and add device-to-parameter links.

### File
`scripts/combine_graphs.py`

### Inputs
- `str_graph.json`: Structural graph from Stage 1
- `fun_graph.json` (or `fun_updated.json`): Functional graph from Stage 2 (or original)

### Outputs
- `comb_graph.json`: Combined graph with all nodes and links

### Merging Strategy

1. **Merge nodes**: Combine all unique node IDs from both graphs (structure takes precedence if node exists in both)
2. **Merge links**: Combine unique edges, deduplicating by (source, target) and optional attributes
3. **Add MOS device-to-parameter links**: For each MOS-like device (e.g., `dev:M2`):
   - Create links: `dev:M2` → `W_M2` and `dev:M2` → `L_M2`
   - These connect device nodes directly to their width and length parameters

### Example Usage

```bash
python scripts/combine_graphs.py \
  --str_graph netlists/diff_amps/75/str_graph.json \
  --fun_graph netlists/diff_amps/75/fun_updated.json \
  --out netlists/diff_amps/75/comb_graph.json
```

Or using directory shortcut:
```bash
python scripts/combine_graphs.py \
  --str_graph netlists/diff_amps/75/ \
  --fun_graph netlists/diff_amps/75/ \
  --out netlists/diff_amps/75/
```

(The script will look for `str_graph.json` and `fun_graph.json` by default, and write `comb_graph.json`)

### Key Functions
- `merge_nodes()`: Combine node lists, avoiding duplicates
- `merge_links()`: Combine edge lists with deduplication
- `add_device_parameter_links()`: Connect MOS devices to W/L parameters
- `is_mos_like()`: Detect MOS device nodes

---

## Stage 4: Build GNN Features

### Purpose
Convert combined graphs into machine-learning-ready representations:
- **Feature matrix (Nx14+)**: Encode node types, sub-categories, and meanings
- **Adjacency matrix (NxN)**: Binary matrix representing edges

### File
`scripts/comb_graph_to_gnn.py`

### Inputs
- `comb_graph.json`: Combined graph

### Outputs
- `comb_graph_gnn.npz`: Compressed NumPy archive containing:
  - `nodes`: Array of node IDs (in order)
  - `features`: NxD feature matrix (float32)
  - `adjacency`: NxN adjacency matrix (uint8, binary)
- `comb_graph_gnn_meta.json`: Metadata with feature dimension, type mappings, meanings

### Feature Encoding

Each node gets a feature vector of dimension D = 6 + 4 + meaning_dim:

#### 1. Node Type (6 dimensions, one-hot)
Position 0: `performance`
Position 1: `sub-structure`
Position 2: `parameter`
Position 3: `net`
Position 4: `device`
Position 5: `terminal`

#### 2. Sub-category (4 dimensions)
Interpretation depends on node type:

**Performance nodes:**
- Position 0: Original (e.g., `Gain`)
- Position 1: Ambiguous variant (`Gain-ambiguous`)
- Position 2: Trade-off variant (`Gain-trade-off`)
- Position 3: Directly-proportional variant (`Gain-directly-proportional`)

**Parameter nodes:**
- Position 0: Original (e.g., `W_M0`)
- Position 1: Directly-proportional variant (`W_M0-directly-proportional`)
- Position 2: Inversely-proportional variant (`W_M0-inversely-proportional`)
- Position 3: Unused

**Device nodes:**
- Position 0: pmos4/pmos
- Position 1: nmos4/nmos
- Positions 2-3: Unused

**Terminal nodes:**
- Position 0: Drain (D)
- Position 1: Gate (G)
- Position 2: Source (S)
- Position 3: Unused

**Sub-structure & Net nodes:**
- All zeros

#### 3. Meaning (meaning_dim dimensions, one-hot)
Interpreted per node type:

**Performance nodes:**
- Dimension = number of unique performance meanings found (e.g., [Gain, CMRR, UGF, Power])
- Each performance node is one-hot encoded by its meaning

**Substructure nodes:**
- Dimension = number of unique substructure types found in the graph
- Each substructure node is one-hot encoded by its type

**Other nodes (parameter, net, device, terminal):**
- All zeros

### Example Usage

```bash
python scripts/comb_graph_to_gnn.py \
  --in netlists/diff_amps/75/comb_graph.json
```

Or to write outputs to a specific directory:
```bash
python scripts/comb_graph_to_gnn.py \
  --in netlists/diff_amps/75/ \
  --out-dir netlists/diff_amps/75/
```

### Output Metadata

The `comb_graph_gnn_meta.json` contains:
```json
{
  "feature_dim": 14,
  "type_order": ["performance", "sub-structure", "parameter", "net", "device", "terminal"],
  "subcat_slots": 4,
  "meaning_dim": 4,
  "performance_meanings": ["Gain", "CMRR", "UGF", "Power"],
  "substructure_types": ["M0-M1 differential pair", "M2-M3 active load current mirror", ...]
}
```

### Loading GNN Data in Python

```python
import numpy as np

# Load the data
data = np.load('netlists/diff_amps/75/comb_graph_gnn.npz', allow_pickle=True)
nodes = data['nodes']
features = data['features']  # shape (N, D)
adjacency = data['adjacency']  # shape (N, N), binary

# Adjacency is symmetric (undirected graph)
# Nodes list gives the mapping from matrix index to node ID
print(f"Number of nodes: {len(nodes)}")
print(f"Feature dimension: {features.shape[1]}")
```

### Key Functions
- `detect_performance_meanings()`: Find all unique performance node IDs
- `detect_substructure_types()`: Find all unique substructure types
- `build_feature_matrix()`: Encode all nodes into Nx(6+4+meaning_dim) matrix
- `build_adjacency()`: Create NxN binary adjacency matrix from edges

---

## Complete Pipeline Example

### Processing a New Circuit

Suppose you have a new differential amplifier circuit in `netlists/diff_amps/NEW_ID/`:
- `NEW_ID.cir`: SPICE netlist
- `fun_graph.json`: Functional graph (parameters and performance metrics)

#### Step 1: Generate Structural Graph
```bash
python get_netlist_to_SG.py \
  --netlist-path netlists/diff_amps/NEW_ID/NEW_ID.cir \
  --output-jsonl netlists/diff_amps/NEW_ID/str_graph.json
```

#### Step 2: Transform Functional Graph
a)First generate query to feed into LLM to get functional graph

```bash
python scripts/generate_fun_graph_prompt.py --circuit netlists/diff_amps/NEW_ID/

```

b) Next, generate query to feed into LLM to get prunable elements

```bash 
python scripts/generate_prune_prompt.py --circuit netlists/diff_amps/NEW_ID/ --out netlists/diff_amps/NEW_ID/prune_prompt.txt
```

c) Next, transform the functional graph to counter edge types.
```bash
python scripts/transform_fun_graph.py \
  --in netlists/diff_amps/NEW_ID/fun_graph.json \
  --out netlists/diff_amps/NEW_ID/fun_updated.json
```

#### Step 3: Combine Graphs
```bash
python scripts/combine_graphs.py \
  --str_graph netlists/diff_amps/NEW_ID/str_graph.json \
  --fun_graph netlists/diff_amps/NEW_ID/fun_updated.json \
  --out netlists/diff_amps/NEW_ID/comb_graph.json
```

#### Step 4: Build GNN Features
```bash
python scripts/comb_graph_to_gnn.py \
  --in netlists/diff_amps/NEW_ID/comb_graph.json
```

#### Batch Processing All Circuits
```bash
for d in netlists/diff_amps/*/; do
  echo "Processing $d"
  python get_netlist_to_SG.py --netlist-path "$d"/*.cir --output-jsonl "$d/str_graph.json"
  python scripts/transform_fun_graph.py --in "$d/fun_graph.json" --out "$d/fun_updated.json"
  python scripts/combine_graphs.py --str_graph "$d/str_graph.json" --fun_graph "$d/fun_updated.json" --out "$d/comb_graph.json"
  python scripts/comb_graph_to_gnn.py --in "$d/comb_graph.json"
done
```

---

## Data Structures

### Structural Graph (str_graph.json)
```json
{
  "directed": false,
  "multigraph": false,
  "graph": {},
  "nodes": [
    {"id": "dev:M2", "type": "device", "device_type": "pmos4"},
    {"id": "term:M2:D", "type": "terminal", "device": "M2", "role": "D"},
    {"id": "net:VOUT1", "type": "net"}
  ],
  "links": [
    {"source": "dev:M2", "target": "term:M2:D"},
    {"source": "term:M2:D", "target": "net:VOUT1"}
  ]
}
```

### Functional Graph (fun_graph.json)
```json
{
  "nodes": [
    {"id": "Gain", "type": "performance"},
    {"id": "W_M0", "type": "parameter"},
    {"id": "M0-M1 differential pair", "type": "sub-structure"}
  ],
  "links": [
    {"source": "W_M0", "target": "Gain", "relation": "directly-proportional"},
    {"source": "W_M0", "target": "M0-M1 differential pair", "relation": "belongs-to"}
  ]
}
```

### Combined Graph (comb_graph.json)
- Merges nodes and links from structural and functional graphs
- Includes all performance, parameter, substructure, device, net, and terminal nodes
- Device-to-parameter links added: `dev:M0` → `W_M0`, `dev:M0` → `L_M0`

### GNN Metadata (comb_graph_gnn_meta.json)
```json
{
  "feature_dim": 14,
  "type_order": ["performance", "sub-structure", "parameter", "net", "device", "terminal"],
  "subcat_slots": 4,
  "meaning_dim": 4,
  "performance_meanings": ["Gain", "CMRR", "UGF", "Power"],
  "substructure_types": [...]
}
```

---

## Dependencies

- **Python 3.7+**
- **Standard library**: `json`, `argparse`, `re`, `os`
- **Optional**: `numpy` (highly recommended for GNN features; falls back to JSON if unavailable)

### Installation
```bash
pip install numpy
```

---

