#!/usr/bin/env python3
"""scripts/visualize_embeddings_umap.py

Extract embeddings from a trained GNN checkpoint and visualize with UMAP.

This script is aligned with the training configs under `gnn_training/config/`.
By default it uses `full_training_diff_amps_comparators_all.yaml` and the
checkpoint in `./checkpoints_full_training/`.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

# Try to import umap; install if not available
try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
    import umap

from gnn_training.models import ContrastiveGINModel
from gnn_training.data import CircuitDataLoader
from gnn_training.perturbations import RemoveKnowledgeNodes


def _lazy_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml  # type: ignore
        return yaml


def load_training_config(config_path: str) -> Dict:
    yaml = _lazy_import_yaml()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def flatten_circuits(circuits_by_family: Dict) -> Tuple[List[str], List[str]]:
    circuit_ids: List[str] = []
    families: List[str] = []
    for family, ids in circuits_by_family.items():
        for cid in ids:
            circuit_ids.append(str(cid))
            families.append(str(family))
    return circuit_ids, families


def try_load_performance_value(comb_graph_path: str, metric_key: str) -> Optional[float]:
    try:
        with open(comb_graph_path, 'r') as f:
            cg = json.load(f)

        # Common schema first
        perf = cg.get('performance')
        if isinstance(perf, dict):
            val = perf.get(metric_key)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                v2 = val.get('value')
                if isinstance(v2, (int, float)):
                    return float(v2)

        # Fallback: some graphs store performances as nodes; scan for numeric fields.
        for n in cg.get('nodes', []):
            if not isinstance(n, dict):
                continue
            name = str(n.get('name', ''))
            ntype = str(n.get('type', ''))
            if metric_key.lower() in name.lower() or metric_key.lower() in ntype.lower():
                for k in ('value', 'val', 'target', 'y'):
                    v = n.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
        return None
    except Exception:
        return None


def _infer_dims_from_checkpoint_state(checkpoint: Dict) -> Dict[str, int]:
    """Infer key model dims directly from checkpoint tensors when possible."""
    state = checkpoint.get('model_state') or {}
    if not isinstance(state, dict):
        return {}

    inferred: Dict[str, int] = {}

    # First GIN MLP Linear weight is [hidden_dims[0], input_dim]
    w = state.get('encoder.gin_layers.0.mlp.0.weight')
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        inferred['hidden0'] = int(w.shape[0])
        inferred['input_dim'] = int(w.shape[1])

    # Projection head final Linear is [projection_dim, embedding_dim]
    # This key should exist for the current ContrastiveGINModel.
    pw = state.get('projection_head.2.weight')
    if isinstance(pw, torch.Tensor) and pw.ndim == 2:
        inferred['projection_dim'] = int(pw.shape[0])
        inferred['embedding_dim'] = int(pw.shape[1])

    return inferred


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_cfg: Optional[Dict] = None,
) -> Tuple[ContrastiveGINModel, Dict]:
    """Load trained model from checkpoint.

    Prefers dimensions from the training YAML, and falls back to inferring
    dimensions from the checkpoint state dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = model_cfg or {}

    inferred = _infer_dims_from_checkpoint_state(checkpoint)

    input_dim = int(model_cfg.get('input_dim', inferred.get('input_dim', 79)))
    hidden_dims = list(model_cfg.get('hidden_dims', [64, 64, 32]))
    embedding_dim = int(model_cfg.get('embedding_dim', inferred.get('embedding_dim', 128)))
    projection_dim = int(model_cfg.get('projection_dim', inferred.get('projection_dim', 128)))
    dropout = float(model_cfg.get('dropout', 0.1))
    use_batch_norm = bool(model_cfg.get('use_batch_norm', True))

    model = ContrastiveGINModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )

    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError:
        inferred = _infer_dims_from_checkpoint_state(checkpoint)
        inferred_input_dim = inferred.get('input_dim')
        inferred_embedding_dim = inferred.get('embedding_dim')
        inferred_projection_dim = inferred.get('projection_dim')
        if inferred_input_dim is None:
            raise
        model = ContrastiveGINModel(
            input_dim=int(inferred_input_dim),
            hidden_dims=hidden_dims,
            embedding_dim=int(inferred_embedding_dim or embedding_dim),
            projection_dim=int(inferred_projection_dim or projection_dim),
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def extract_embeddings(model: ContrastiveGINModel, 
                       circuit_ids: List[str],
                       families: List[str],
                       data_dir: str,
                       device: torch.device, 
                       remove_knowledge_graph: bool = False,) -> Tuple[np.ndarray, List[Dict]]:
    """Extract embeddings for all circuits."""
    embeddings = []
    metadata = []
    
    with torch.no_grad():
        for circuit_id, family in zip(circuit_ids, families):
            try:
                loader = CircuitDataLoader(circuit_id, f"{data_dir}/{family}")
                graph = loader.get_graph()

                # Remove Knoowledge Graph (Using only SG in test-time)
                if remove_knowledge_graph:
                    graph = RemoveKnowledgeNodes(graph).apply()

                # Convert to tensors
                features = torch.tensor(graph['features'], dtype=torch.float32, device=device)
                adjacency = torch.tensor(graph['adjacency'], dtype=torch.float32, device=device)
                
                # Get embedding from encoder (before projection head)
                embedding = model.encode(features, adjacency).cpu().numpy()
                embeddings.append(embedding)
                
                # Store metadata
                metadata.append({
                    'circuit_id': circuit_id,
                    'family': family,
                    'embedding_norm': float(np.linalg.norm(embedding))
                })
                
                print(f"✓ Extracted embedding for {family}/{circuit_id}")
            except Exception as e:
                print(f"✗ Failed to extract embedding for {family}/{circuit_id}: {e}")
    
    embeddings_array = np.array(embeddings)  # [num_circuits, embedding_dim]
    return embeddings_array, metadata


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 5, min_dist: float = 0.1) -> np.ndarray:
    """Compute UMAP projection."""
    print(f"\nComputing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"UMAP shape: {umap_embeddings.shape}")
    return umap_embeddings


def plot_embeddings_by_family(umap_emb: np.ndarray, metadata: List[Dict], output_path: str):
    """Plot UMAP embeddings colored by family."""
    families = [m['family'] for m in metadata]
    unique_families = sorted(set(families))

    # Stable, readable colors for common families; fall back to a palette for others.
    base_colors = {
        'diff_amps': '#1f77b4',
        'comparators': '#ff7f0e',
        'LDO': '#2ca02c',
        'ldo': '#2ca02c',
        'op-amp': '#d62728',
        'opamp': '#d62728',
    }
    palette = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    family_to_color: Dict[str, str] = {}
    palette_idx = 0
    for fam in unique_families:
        if fam in base_colors:
            family_to_color[fam] = base_colors[fam]
        else:
            family_to_color[fam] = palette[palette_idx % len(palette)]
            palette_idx += 1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for family in unique_families:
        mask = np.array([f == family for f in families])
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], 
                  label=family, s=150, alpha=0.7, color=family_to_color[family])
    
    # Add circuit ID labels
    for i, m in enumerate(metadata):
        ax.annotate(m['circuit_id'], (umap_emb[i, 0], umap_emb[i, 1]), 
                   fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('GNN Embeddings - UMAP Projection (by Family)', fontsize=14, fontweight='bold')
    ax.legend(title='Circuit Family', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def plot_embeddings_by_performance(umap_emb: np.ndarray, metadata: List[Dict], 
                                   data_dir: str, output_path: str,
                                   metric_key: str = 'Gain'):
    """Plot UMAP embeddings colored by a performance metric."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    values: List[float] = []
    for m in metadata:
        comb_graph_path = f"{data_dir}/{m['family']}/{m['circuit_id']}/comb_graph.json"
        v = try_load_performance_value(comb_graph_path, metric_key)
        values.append(float(v) if v is not None else float('nan'))

    values_arr = np.array(values, dtype=float)
    # Handle missing values
    if np.all(np.isnan(values_arr)):
        values_arr = np.zeros_like(values_arr)
    else:
        nan_mask = np.isnan(values_arr)
        if np.any(nan_mask):
            values_arr[nan_mask] = np.nanmedian(values_arr)

    scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=values_arr, cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add circuit ID labels
    for i, m in enumerate(metadata):
        ax.annotate(m['circuit_id'], (umap_emb[i, 0], umap_emb[i, 1]), 
                   fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'GNN Embeddings - UMAP Projection ({metric_key} coloring)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(metric_key, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def save_embeddings_json(embeddings: np.ndarray, metadata: List[Dict], output_path: str):
    """Save embeddings and metadata to JSON for interactive visualization."""
    data = {
        'embeddings': embeddings.tolist(),
        'metadata': metadata,
        'embedding_dim': embeddings.shape[1],
        'num_circuits': embeddings.shape[0]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize circuit embeddings with UMAP.')
    parser.add_argument('--config', default='gnn_training/config/full_training_diff_amps_comparators_all.yaml',
                        help='Training YAML used to pick circuits and model dims.')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path. If omitted, uses training.checkpoint_dir + last epoch.')
    parser.add_argument('--data-dir', default='netlists',
                        help='Netlists root (contains family folders).')
    parser.add_argument('--out-dir', default='umap_results_sg_vs_skg',
                        help='Output directory for plots/json.')
    parser.add_argument('--metric', default='Gain',
                        help='Performance metric key for coloring (e.g., Gain, UGB, PM).')
    parser.add_argument('--n-neighbors', type=int, default=8)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})

    device = torch.device(args.device)
    data_dir = args.data_dir
    output_dir = Path(args.out_dir)
    output_dir.mkdir(exist_ok=True)

    circuits_cfg = cfg.get('data', {}).get('circuits', {})
    circuit_ids, families = flatten_circuits(circuits_cfg)
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = str(train_cfg.get('checkpoint_dir', './checkpoints_full_training'))
        epochs = int(train_cfg.get('epochs', 20))
        checkpoint_path = f"{ckpt_dir}/checkpoint_epoch_{epochs:03d}.pt"

    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint_info = load_checkpoint(checkpoint_path, device, model_cfg=model_cfg)

    print(f"✓ Model loaded (epoch {checkpoint_info['epoch']})")
    
    # remove knowledge graph in test-time
    remove_knowledge_graph = cfg.get('testing', {}).get('remove_knowledge_graph', False)
    if remove_knowledge_graph:
        print("Test-time setting: Removing knowledge graph from all circuits for embedding extraction.")

    print(f"\nExtracting embeddings for {len(circuit_ids)} circuits...")
    embeddings, metadata = extract_embeddings(model, circuit_ids, families, data_dir, device, remove_knowledge_graph=remove_knowledge_graph)
    print(f"✓ Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    print(f"\nNormalizing embeddings...")
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    print(f"\nComputing UMAP projection...")
    umap_embeddings = compute_umap(embeddings_norm, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    
    print(f"\nGenerating visualizations...")
    plot_embeddings_by_family(umap_embeddings, metadata, str(output_dir / 'umap_by_family.png'))
    plot_embeddings_by_performance(
        umap_embeddings,
        metadata,
        data_dir,
        str(output_dir / 'umap_by_performance.png'),
        metric_key=args.metric,
    )
    
    print(f"\nSaving embeddings data...")
    save_embeddings_json(umap_embeddings, metadata, str(output_dir / 'umap_embeddings.json'))
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EMBEDDING SUMMARY")
    print("="*60)
    for i, m in enumerate(metadata):
        print(f"{m['family']:12s} {m['circuit_id']:6s} | norm: {m['embedding_norm']:.4f}")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
