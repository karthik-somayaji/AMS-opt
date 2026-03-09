# Embeddings pipeline (LLMBO)

This describes how LLMBO goes from stored graph files to GNN embeddings.

## 0) Embedding modes: `skg` vs `sg`

LLMBO can embed two different graph inputs using the **same trained encoder**:

- `--gnn_embedding_mode skg` (default): encode the full combined graph (Structural Graph + Knowledge Graph).
- `--gnn_embedding_mode sg`: strip knowledge nodes first (via `gnn_training.perturbations.RemoveKnowledgeNodes`) and encode the structural-only graph.

Important:
- Your `--embeddings_json` must match the mode (SKG reference embeddings vs SG reference embeddings), otherwise similarity search is meaningless.
- In `sg` mode, encoding `comb_graph_gnn.npz` requires the NPZ to contain a `nodes` array (used to identify knowledge nodes).

## 1) Anchor embedding (amp2 / FC / comp / ldo)

**Inputs (stored on disk)**
- Anchor graph (features+adjacency):
  - `LLMBO/amp2_ati_new/comb_graph_gnn.npz`
  - `LLMBO/FC_ati_new/comb_graph_gnn.npz`
  - `LLMBO/comp_ati_new/comb_graph_gnn.npz`
  - `LLMBO/ldo_ati_new/comb_graph_gnn.npz`

**Model checkpoint (stored on disk)**
- Trained encoder weights: `checkpoints_full_training/best_model.pt` (default)

**Code path (in `LLMBO/llmbo.py`)**
1. `_maybe_prepare_llmbo_anchor_embedding(args)` runs when `--history 1` and `--target_id` is one of `{amp2, FC, comp, ldo}`.
2. Loads the trained model via `_load_gnn_model(checkpoint_path=..., device=...)`.
3. Encodes the NPZ via `_encode_comb_graph_npz_mode(model, npz_path=..., device=..., embedding_mode=args.gnn_embedding_mode)`:
   - `np.load(npz_path)["features"], np.load(npz_path)["adjacency"]`
  - If `--gnn_embedding_mode sg`: removes knowledge nodes first, then encodes
  - `model.encode(feat_t, adj_t)` → embedding vector `emb` (1D)
4. Stores the embedding **in memory** on `args.target_anchor_embedding` (a `list[float]`).

**Output**
- Anchor embedding vector is *not* written to a dedicated file by default; it is computed on demand.

## 2) Reference embeddings (netlists dataset)

**Metadata that selects which circuits to embed (stored on disk)**
- `umap_results_full/umap_embeddings.json`
  - Must contain `metadata: [{"circuit_id": "...", "family": "..."}, ...]`

**Where reference netlists live**
- `netlists/<family>/...`

**Code path (in `LLMBO/llmbo.py`)**
1. `_maybe_prepare_llmbo_anchor_embedding(args)` calls `_ensure_reference_embeddings_json(...)`.
2. `_ensure_reference_embeddings_json(...)`:
   - Reads `umap_results_full/umap_embeddings.json` for the `metadata` list
   - For each `{circuit_id, family}`:
     - Loads the graph via `gnn_training.data.CircuitDataLoader(circuit_id, fam_dir).get_graph()`
  - If `--gnn_embedding_mode sg`: applies `RemoveKnowledgeNodes` before encoding
  - Encodes with the same `model.encode(feat_t, adj_t)` as the anchor pipeline

**Output (stored on disk)**
- Raw GNN reference embeddings JSON:
  - `umap_results_full/gnn_embeddings.json`
  - Format: `{ "embeddings": [[...], ...], "metadata": [{"circuit_id":..., "family":...}, ...] }`

**SG reference embeddings (stored on disk)**
- If `--gnn_embedding_mode sg` and you leave `--embeddings_json` at its default value, LLMBO will automatically switch the output to:
  - `umap_results_full/gnn_embeddings_sg.json`

Notes on caching/refresh:
- `_ensure_reference_embeddings_json(...)` is a “create-if-missing” helper. If you want to regenerate embeddings with a different checkpoint/config, delete the existing JSON first.

## 3) Quickstart commands

Generate (or reuse) reference embeddings via LLMBO (recommended if you already run LLMBO):

```bash
# SKG (default): will create umap_results_full/gnn_embeddings.json if missing
PYTHONPATH=$PWD python -u LLMBO/llmbo.py \
  --history 1 \
  --target_id amp2 \
  --gnn_checkpoint checkpoints_full_training/best_model.pt \
  --gnn_device cpu \
  --gnn_embedding_mode skg

# SG: will create umap_results_full/gnn_embeddings_sg.json if missing
PYTHONPATH=$PWD python -u LLMBO/llmbo.py \
  --history 1 \
  --target_id amp2 \
  --gnn_checkpoint checkpoints_sg_vs_skg/best_model.pt \
  --gnn_device cpu \
  --gnn_embedding_mode sg
```

Generate embeddings (and SG-vs-SKG diagnostics/plots) via the analysis script:

```bash
PYTHONPATH=$PWD python -u gnn_training/scripts/analyze_sg_skg_embeddings.py \
  --checkpoint checkpoints_sg_vs_skg/best_model.pt \
  --netlists_root netlists \
  --out_dir umap_results_full/sg_skg_analysis \
  --export_llmbo_json

# This writes (among other artifacts):
# - umap_results_full/sg_skg_analysis/gnn_embeddings_sg_all_netlists.json
# - umap_results_full/sg_skg_analysis/gnn_embeddings_skg_all_netlists.json
```

If you want LLMBO to use the exported SG JSON explicitly:

```bash
PYTHONPATH=$PWD python -u LLMBO/llmbo.py \
  --history 1 \
  --target_id amp2 \
  --gnn_embedding_mode sg \
  --embeddings_json umap_results_full/gnn_embeddings_sg.json
```

## Notes
- Anchor and reference embeddings are in the same raw GNN embedding space (same embedding dimension `D`).
- You can override paths via CLI args in `LLMBO/llmbo.py`:
  - `--gnn_checkpoint`, `--gnn_device`, `--embeddings_json`, `--reference_metadata_json`
