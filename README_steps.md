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
