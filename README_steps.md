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
python3 scripts/generate_global_meanings.py --families comparators diff_amps LDO op-amp --rules scripts/meaning_aliases.json --report netlists/_meaning_curation_report.json
```

**Recommended (robust) flow: alias rules + curation report**

To avoid a blow-up of near-duplicate substructures/performance names, use the manual alias rules file `scripts/meaning_aliases.json`.

1) Dry-run with a curation report (no outputs written):

```bash
python3 scripts/generate_global_meanings.py \
  --families comparators diff_amps LDO op-amp \
  --rules scripts/meaning_aliases.json \
  --dry-run \
  --report netlists/_meaning_curation_report.json
```

2) Edit `scripts/meaning_aliases.json` to club/merge names based on domain knowledge.

3) Generate the final global lists:

```bash
python3 scripts/generate_global_meanings.py \
  --families comparators diff_amps LDO op-amp \
  --rules scripts/meaning_aliases.json \
  --report netlists/_meaning_curation_report.json
```

This produces:
- `netlists/_performance_meanings.json`
- `netlists/_substructures_ordered.json`
- `netlists/_substructures_mapping.json` (canonical -> raw examples)
- `netlists/_meaning_curation_report.json` (raw->canonical + counts to help refine rules)

b) Get features
```bash
python scripts/comb_graph_to_gnn.py \
  --in netlists/diff_amps/NEW_ID/comb_graph.json
```

```
python3 scripts/comb_graph_to_gnn_global.py \
    --in LLMBO/$c/ \
    --out-dir LLMBO/$c/ \
    --netlists-root netlists \
    --maxima-source training-meta
```


Or batch process using:
```
for fam in netlists/*/; do
  famname="$(basename "$fam")"
  # skip global json files/dirs like netlists/_performance_meanings.json
  [[ "$famname" == _* ]] && continue

  echo "=== Family: $famname ==="
  count=0

  for d in "$fam"*/; do
    [[ -d "$d" ]] || continue
    id="$(basename "$d")"
    cg="$d/comb_graph.json"
    alt="$d/${id}_comb_graph.json"

    if [[ -f "$cg" ]]; then
      python3 scripts/comb_graph_to_gnn.py --in "$cg"
      ((count+=1))
    elif [[ -f "$alt" ]]; then
      python3 scripts/comb_graph_to_gnn.py --in "$alt"
      ((count+=1))
    else
      echo "Skipping (no comb_graph): $d"
    fi
  done

  echo "Processed $count circuits in $famname"
done
```

OR

```
for fam in netlists/*/; do
  famname="$(basename "$fam")"
  [[ "$famname" == _* ]] && continue

  for d in "$fam"*/; do
    [[ -d "$d" ]] || continue
    python3 scripts/comb_graph_to_gnn_global.py \
      --in "$d" \
      --out-dir "$d" \
      --netlists-root netlists \
      --maxima-source training-meta
  done
done
```

#####################


## Final training & Visualization

```python3 -m gnn_training.scripts.train \
  --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml
```

```
python3 scripts/visualize_embeddings_umap.py   --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml   --device cpu
```


```
 /home/karthik/miniconda3/envs/analog-rep/bin/python scripts/analyze_four_test_circuits_embeddings.py \
  --llmbo-circuits amp2:LLMBO/amp2_ati_new FC:LLMBO/FC_ati_new comp:LLMBO/comp_ati_new ldo:LLMBO/ldo_ati_new \
  --out-dir umap_results_four
```

##########

## Optimization

cd /home/karthik/sim_clean/AMS-opt && /home/karthik/miniconda3/envs/analog-rep/bin/python LLMBO/llmbo.py --history 1 --related_mode topk --related_k 3 --target_id amp2

cd /home/karthik/sim_clean/AMS-opt && /home/karthik/miniconda3/envs/analog-rep/bin/python LLMBO/llmbo.py --history 1 --related_mode random_family --related_k 3 --target_id amp2