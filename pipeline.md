# AMS-opt Pipeline (Netlist → Graphs → GNN Training)

This document is a practical, repo-specific guide to the end-to-end pipeline:

`SPICE netlists` → `structural graph` + `functional/KG graph` → `combined graph` → `GNN-ready NPZ` → `contrastive training` → `embedding visualizations`

---

## Directory Structure (What lives where)

```
AMS-opt/
  get_netlist_to_SG.py                # Stage 1: netlist → structural graph (SG)
  scripts/
    generate_fun_graph_prompt.py      # Prompts for LLM to produce fun_graph.json
    generate_prune_prompt.py          # Prompts for prunable elements (optional)
    transform_fun_graph.py            # Stage 2: expand variants + collapse relations → fun_updated.json
    combine_graphs.py                 # Stage 3: SG + KG merge + dev→(W,L) links → comb_graph.json
    collect_substructures.py          # Pre-pass: canonicalize substructures + performance meanings
    comb_graph_to_gnn.py              # Stage 4: comb_graph.json → comb_graph_gnn.npz + meta

    smoke_test_train.py               # Convenience script to run a small training
    visualize_embeddings_umap.py      # Extract embeddings + make UMAP plots
    create_interactive_umap.py        # Convert UMAP JSON → interactive Plotly HTML
    generate_presentation.py          # Generates a PowerPoint deck (PPTX) about the pipeline

  netlists/
    diff_amps/
      <circuit_id>/
        *.cir                         # SPICE netlist
        fun_graph.json                # LLM-produced functional/KG graph (input)
        fun_updated.json              # Stage 2 output (variants + connects)
        str_graph.json                # Stage 1 output (structural graph)
        comb_graph.json               # Stage 3 output (combined)
        comb_graph_gnn.npz            # Stage 4 output (nodes/features/adjacency)
        comb_graph_gnn_meta.json      # Stage 4 metadata (feature layout + mappings)
      _substructures_ordered.json     # Canonical substructure list for the family
      _performance_meanings.json      # Canonical base performance metrics list

    comparators/
      <circuit_id>/
        ... same artifacts as above ...
      _substructures_ordered.json
      _performance_meanings.json

    _substructures_ordered.json       # (optional) global canonical list (if generated)
    _performance_meanings.json        # (optional) global canonical list (if generated)

  gnn_training/
    config/                           # Training YAMLs (default/smoke/full/etc.)
    data/                             # Loaders for comb_graph_gnn.npz
    models/                           # GIN encoder + projection head
    losses/                           # NT-Xent, weighted NT-Xent, consistency loss
    pairs/                            # positive/negative sampling and hard negative mining
    training/                         # Trainer/Validator/EarlyStopping/Scheduler
    scripts/
      train.py                        # Main entry: `python -m gnn_training.scripts.train --config ...`

  checkpoints_*/                      # Saved models and configs
  logs_*/                             # TensorBoard event files
  umap_results_*/                     # UMAP plots/JSON/HTML
```

---

## Data Contracts (Files produced at each stage)

### Stage 1 — Structural Graph
- **Input**: SPICE netlist (`*.cir`)
- **Script**: `get_netlist_to_SG.py`
- **Output**: `str_graph.json`

Structural graph node types:
- `device`: `dev:M0`, `dev:R1` (has `device_type` like `nmos4`, `pmos4`, `resistor`)
- `terminal`: `term:M0:D`, `term:M0:G`, `term:M0:S`
- `net`: `net:VDD`, `net:VOUT1`, ...

Structural graph edges (examples):
- MOS: `dev:M0 → term:M0:D/G/S` and `term:M0:<role> → net:<name>`
- Non-MOS: `dev:R0 → net:<name>`


### Stage 2 — Functional / Knowledge Graph (KG)
- **Input**: `fun_graph.json` (LLM-produced)
- **Script**: `scripts/transform_fun_graph.py`
- **Output**: `fun_updated.json`

What happens in Stage 2:
- Creates **variant nodes** for relations:
  - performance: `Gain-ambiguous`, `Gain-trade-off`, `Gain-directly-proportional`, etc.
  - parameter: `W_M0-directly-proportional`, `W_M0-inversely-proportional`, etc.
- Collapses relation-typed edges into uniform edges with relation `connects` between the right variants.


### Stage 3 — Combined Graph
- **Inputs**: `str_graph.json` and `fun_updated.json` (or `fun_graph.json`)
- **Script**: `scripts/combine_graphs.py`
- **Output**: `comb_graph.json`

What happens in Stage 3:
- Merges nodes and deduplicates links across SG and KG.
- Adds MOS device-to-parameter edges:
  - `dev:Mk → W_Mk` and `dev:Mk → L_Mk`


### Stage 4 — GNN-ready Features
- **Input**: `comb_graph.json`
- **Script**: `scripts/comb_graph_to_gnn.py`
- **Outputs**:
  - `comb_graph_gnn.npz` (arrays: `nodes`, `features`, `adjacency`)
  - `comb_graph_gnn_meta.json` (feature layout + mappings)

Feature layout (as implemented in `scripts/comb_graph_to_gnn.py`):
- **Type one-hot (6 dims)**: `[performance, sub-structure, parameter, net, device, terminal]`
- **Subcategory slots (4 dims)**: meaning depends on node type
  - performance: `[orig, ambiguous, trade-off, directly-proportional]`
  - parameter: `[orig, directly-proportional, inversely-proportional, unused]`
  - device: `[pmos, nmos, unused, unused]`
  - terminal: `[D, G, S, unused]`
- **Meaning block**: concatenated blocks `[performance_meaning | substructure_meaning]`
  - terminal/net/device/parameter do not populate meaning
  - performance variants map to **base** metrics for meaning (e.g., `Gain-trade-off` → meaning=`Gain`)
  - substructures are normalized to canonical strings before one-hot

---

## Pre-pass (Canonicalize substructures & performance meanings)

Run once per family after `comb_graph.json` files exist:

```bash
python3 scripts/collect_substructures.py --family diff_amps --out netlists/diff_amps/_substructures_ordered.json
python3 scripts/collect_substructures.py --family comparators --out netlists/comparators/_substructures_ordered.json
```

This writes:
- `<family>/_substructures_ordered.json`
- `<family>/_performance_meanings.json`

These canonical lists are then used consistently when creating feature vectors.

---

## Typical Per-Circuit Run (Manual)

From repo root (`AMS-opt/`):

```bash
# Stage 1: netlist → structural graph
python3 get_netlist_to_SG.py \
  --netlist-path netlists/diff_amps/77/77.cir \
  --output-jsonl netlists/diff_amps/77/str_graph.json

# Stage 2: functional graph (LLM output) → expanded + collapsed graph
python3 scripts/transform_fun_graph.py \
  --in netlists/diff_amps/77/fun_graph.json \
  --out netlists/diff_amps/77/fun_updated.json

# Stage 3: merge SG + KG → combined
python3 scripts/combine_graphs.py \
  --str_graph netlists/diff_amps/77/str_graph.json \
  --fun_graph netlists/diff_amps/77/fun_updated.json \
  --out netlists/diff_amps/77/comb_graph.json

# Stage 4: combined → NPZ features
python3 scripts/comb_graph_to_gnn.py --in netlists/diff_amps/77/comb_graph.json
```

---

## Batch Run (All circuits in a family)

```bash
# (optional but recommended) canonicalize substructures first
python3 scripts/collect_substructures.py --family diff_amps --out netlists/diff_amps/_substructures_ordered.json

for d in netlists/diff_amps/*/; do
  if [ -f "$d"/*.cir ] && [ -f "$d/fun_graph.json" ]; then
    echo "Processing $d"
    python3 get_netlist_to_SG.py --netlist-path "$d"/*.cir --output-jsonl "$d/str_graph.json"
    python3 scripts/transform_fun_graph.py --in "$d/fun_graph.json" --out "$d/fun_updated.json"
    python3 scripts/combine_graphs.py --str_graph "$d/str_graph.json" --fun_graph "$d/fun_updated.json" --out "$d/comb_graph.json"
    python3 scripts/comb_graph_to_gnn.py --in "$d/comb_graph.json"
  fi
done
```

---

## Training (Contrastive GNN)

Main entrypoint:

```bash
python3 -m gnn_training.scripts.train --config gnn_training/config/full_training_config.yaml
```

Key components:
- **Model**: `gnn_training/models/gin_model.py` (`ContrastiveGINModel` = GIN encoder + projection head)
- **Losses**:
  - `gnn_training/losses/nt_xent_loss.py`: `NTXentLoss`, `WeightedNTXentLoss`
  - `gnn_training/losses/consistency_loss.py`: `ConsistencyLoss` (margin) and `ConsistencyLossAlternative` (MSE)
- **Sampler/miner**:
  - `gnn_training/pairs/*`: multi-circuit sampling + overlap-based hard negatives

Outputs:
- checkpoints: `checkpoints_*/checkpoint_epoch_XXX.pt`, `best_model.pt`
- logs: `logs_*/events.out.tfevents.*` (TensorBoard)

---

## Visualization (UMAP)

```bash
python3 scripts/visualize_embeddings_umap.py
python3 scripts/create_interactive_umap.py
```

Produces:
- `umap_results_full/umap_by_family.png`
- `umap_results_full/umap_by_performance.png`
- `umap_results_full/umap_embeddings.json`
- `umap_results_full/umap_interactive.html`

---

## Notes / Common Pitfalls

- If you run training in a clean environment, you may need `pyyaml`, `torch`, `tensorboard`, `umap-learn`, `plotly`.
- Prefer module invocation for training to keep imports clean:
  - `python3 -m gnn_training.scripts.train --config ...`
