## Complete Pipeline Example

### Pre-Processing: Collect Unique Substructures (Once per Family)

Before processing any circuits in a family, scan all `comb_graph.json` files to collect and normalize substructure types. This generates a canonical ordered list used by `comb_graph_to_gnn.py`:

```bash
python scripts/collect_substructures.py \
  --family diff_amps \
  --out netlists/diff_amps/_substructures_ordered.json
```

This creates `netlists/diff_amps/_substructures_ordered.json` with normalized, globally-ordered substructure types (e.g., `differential pair`, `active load`, `bias`, etc.). Do this once after you have all `comb_graph.json` files for the family.

---

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

a) Collect sub-structures :

```bash
python3 scripts/collect_substructures.py --family comparators --out netlists/comparators/_substructures_ordered.json
```

```
python3 scripts/generate_global_meanings.py
```

b) Get features
```bash
python scripts/comb_graph_to_gnn.py \
  --in netlists/diff_amps/NEW_ID/comb_graph.json
```

#### Batch Processing All Circuits

**Important**: Before running batch processing, ensure you've run `collect_substructures.py` for your family:

```bash
# Once per family: collect canonical substructures
python scripts/collect_substructures.py --family diff_amps --out netlists/diff_amps/_substructures_ordered.json

# Then batch process all circuits
for d in netlists/diff_amps/*/; do
  echo "Processing $d"
  python get_netlist_to_SG.py --netlist-path "$d"/*.cir --output-jsonl "$d/str_graph.json"
  python scripts/transform_fun_graph.py --in "$d/fun_graph.json" --out "$d/fun_updated.json"
  python scripts/combine_graphs.py --str_graph "$d/str_graph.json" --fun_graph "$d/fun_updated.json" --out "$d/comb_graph.json"
  python scripts/comb_graph_to_gnn.py --in "$d/comb_graph.json"
done

#### Batch: Build GNN Features for `diff_amps` + `comparators`

This runs `comb_graph_to_gnn.py` for every circuit that has a `comb_graph.json` and writes outputs in-place (into each circuit directory).

```bash
cd /home/karthik/sim_clean/AMS-opt

for fam in diff_amps comparators; do
  echo "=== Family: $fam ==="
  for d in netlists/$fam/*/; do
    id="$(basename "$d")"
    cg="$d/comb_graph.json"
    alt="$d/${id}_comb_graph.json"
    if [ -f "$cg" ]; then
      echo "Processing $cg"
      python3 scripts/comb_graph_to_gnn.py --in "$cg"
    elif [ -f "$alt" ]; then
      echo "Processing $alt"
      python3 scripts/comb_graph_to_gnn.py --in "$alt"
    else
      echo "Skipping (missing comb_graph.json / ${id}_comb_graph.json): $d"
    fi
  done
done
```
```

---
#####################

cd /home/karthik/sim_clean/AMS-opt && python3 -m gnn_training.scripts.train --config gnn_training/config/full_training_config.yaml 2>&1 | tail -150

python3 -m gnn_training.scripts.train --config gnn_training/config/full_training_config.yaml

cd /home/karthik/sim_clean/AMS-opt && python3 scripts/visualize_embeddings_umap.py 2>&1

#####################
python3 -m gnn_training.scripts.train --config gnn_training/config/full_training_diff_amps_comparators_all.yaml


python3 scripts/visualize_embeddings_umap.py --n-neighbors 12 --min-dist 0.05
python3 scripts/visualize_embeddings_umap.py --metric UGB