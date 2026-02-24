# Embeddings pipeline (LLMBO)

This describes how LLMBO goes from stored graph files to GNN embeddings.

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
3. Encodes the NPZ via `_encode_comb_graph_npz(model, npz_path, device)`:
   - `np.load(npz_path)["features"], np.load(npz_path)["adjacency"]`
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
     - Encodes with the same `model.encode(feat_t, adj_t)` as the anchor pipeline

**Output (stored on disk)**
- Raw GNN reference embeddings JSON:
  - `umap_results_full/gnn_embeddings.json`
  - Format: `{ "embeddings": [[...], ...], "metadata": [{"circuit_id":..., "family":...}, ...] }`

## Notes
- Anchor and reference embeddings are in the same raw GNN embedding space (same embedding dimension `D`).
- You can override paths via CLI args in `LLMBO/llmbo.py`:
  - `--gnn_checkpoint`, `--gnn_device`, `--embeddings_json`, `--reference_metadata_json`
