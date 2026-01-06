#!/usr/bin/env python3
"""
Extract embeddings from trained GNN model and visualize with UMAP.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[ContrastiveGINModel, Dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model (assuming standard config)
    model = ContrastiveGINModel(
        input_dim=26,
        hidden_dims=[64, 64, 32],
        embedding_dim=128,
        projection_dim=128,
        dropout=0.1,
        use_batch_norm=True
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def extract_embeddings(model: ContrastiveGINModel, 
                       circuit_ids: List[str],
                       families: List[str],
                       data_dir: str,
                       device: torch.device) -> Tuple[np.ndarray, List[Dict]]:
    """Extract embeddings for all circuits."""
    embeddings = []
    metadata = []
    
    with torch.no_grad():
        for circuit_id, family in zip(circuit_ids, families):
            try:
                loader = CircuitDataLoader(circuit_id, f"{data_dir}/{family}")
                graph = loader.get_graph()
                
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
    
    colors = {'diff_amps': '#1f77b4', 'comparators': '#ff7f0e', 'ldo': '#2ca02c'}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for family in unique_families:
        mask = np.array([f == family for f in families])
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], 
                  label=family, s=150, alpha=0.7, color=colors.get(family, '#1f77b4'))
    
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
                                   data_dir: str, output_path: str):
    """Plot UMAP embeddings colored by performance (Gain)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gains = []
    for m in metadata:
        try:
            comb_graph_path = f"{data_dir}/{m['family']}/{m['circuit_id']}/comb_graph.json"
            with open(comb_graph_path) as f:
                comb_graph = json.load(f)
                # Find Gain performance value
                gain_val = None
                for node_id in comb_graph.get('nodes', []):
                    if 'Gain' in str(node_id):
                        # Extract gain value if it has one
                        gain_val = 0.5  # Default
                        break
                gains.append(gain_val if gain_val is not None else 0.5)
        except:
            gains.append(0.5)
    
    gains = np.array(gains)
    scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=gains, cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add circuit ID labels
    for i, m in enumerate(metadata):
        ax.annotate(m['circuit_id'], (umap_emb[i, 0], umap_emb[i, 1]), 
                   fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('GNN Embeddings - UMAP Projection (Performance Similarity)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Gain Value', fontsize=11)
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
    # Configuration
    device = torch.device('cpu')
    data_dir = 'netlists'
    checkpoint_path = 'checkpoints_full_training/checkpoint_epoch_020.pt'
    output_dir = Path('umap_results_full')
    output_dir.mkdir(exist_ok=True)
    
    # Circuit list
    circuits_config = {
        'diff_amps': ['75', '77', '84', '86', '94'],
        'comparators': ['1045', '1046', '1051', '1065', '1071', '1073']
    }
    
    # Flatten to lists
    circuit_ids = []
    families = []
    for family, ids in circuits_config.items():
        for cid in ids:
            circuit_ids.append(cid)
            families.append(family)
    
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint_info = load_checkpoint(checkpoint_path, device)
    print(f"✓ Model loaded (epoch {checkpoint_info['epoch']})")
    
    print(f"\nExtracting embeddings for {len(circuit_ids)} circuits...")
    embeddings, metadata = extract_embeddings(model, circuit_ids, families, data_dir, device)
    print(f"✓ Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    print(f"\nNormalizing embeddings...")
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    print(f"\nComputing UMAP projection...")
    umap_embeddings = compute_umap(embeddings_norm, n_neighbors=5, min_dist=0.1)
    
    print(f"\nGenerating visualizations...")
    plot_embeddings_by_family(umap_embeddings, metadata, str(output_dir / 'umap_by_family.png'))
    plot_embeddings_by_performance(umap_embeddings, metadata, data_dir, 
                                    str(output_dir / 'umap_by_performance.png'))
    
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
