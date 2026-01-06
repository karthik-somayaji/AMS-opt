#!/usr/bin/env python3
"""
Visualize learned embeddings using UMAP.
Extracts embeddings from the trained GNN model and creates interactive plots.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_training.models import ContrastiveGINModel
from gnn_training.data import MultiCircuitDataLoader


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[ContrastiveGINModel, dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model (requires knowing the config)
    config_path = Path(checkpoint_path).parent / 'config.yaml'
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = ContrastiveGINModel(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        embedding_dim=config['model']['embedding_dim'],
        projection_dim=config['model']['projection_dim'],
        dropout=config['model'].get('dropout', 0.1),
        use_batch_norm=config['model'].get('use_batch_norm', True),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, config


def extract_embeddings(model: ContrastiveGINModel, loader: MultiCircuitDataLoader,
                      circuit_specs: Dict[str, List], data_dir: str,
                      device: torch.device) -> Tuple[np.ndarray, List[str], List[str]]:
    """Extract embeddings for all circuits."""
    embeddings = []
    circuit_labels = []
    family_labels = []
    
    print("Extracting embeddings...")
    
    with torch.no_grad():
        for family, circuit_ids in circuit_specs.items():
            for circuit_id in circuit_ids:
                circuit_path = Path(data_dir) / family / str(circuit_id)
                
                # Load NPZ file
                npz_file = circuit_path / 'comb_graph_gnn.npz'
                if not npz_file.exists():
                    print(f"  Skipping {family}/{circuit_id} - NPZ not found")
                    continue
                
                data = np.load(npz_file)
                features = torch.tensor(data['features'], dtype=torch.float32, device=device)
                adjacency = torch.tensor(data['adjacency'], dtype=torch.float32, device=device)
                
                # Encode
                embedding = model.encode(features, adjacency)  # [embedding_dim]
                embeddings.append(embedding.cpu().numpy())
                circuit_labels.append(f"{family}_{circuit_id}")
                family_labels.append(family)
                
                print(f"  Extracted: {family}/{circuit_id}")
    
    embeddings = np.array(embeddings)  # [num_circuits, embedding_dim]
    return embeddings, circuit_labels, family_labels


def apply_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Apply UMAP dimensionality reduction."""
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed. Install with:")
        print("  pip install umap-learn")
        sys.exit(1)
    
    print(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_jobs=-1)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    return umap_embeddings


def plot_embeddings(umap_embeddings: np.ndarray, circuit_labels: List[str],
                   family_labels: List[str], output_dir: str = '.'):
    """Create visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Color palette
    families = sorted(set(family_labels))
    colors = {fam: c for fam, c in zip(families, sns.color_palette("husl", len(families)))}
    
    # Plot 1: UMAP colored by family
    fig, ax = plt.subplots(figsize=(10, 8))
    for fam in families:
        mask = np.array(family_labels) == fam
        ax.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1],
                  label=fam, s=200, alpha=0.7, color=colors[fam], edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel("UMAP 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight='bold')
    ax.set_title("Circuit Embeddings (UMAP) - Colored by Family", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_by_family.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'umap_by_family.png'}")
    plt.close()
    
    # Plot 2: UMAP with circuit IDs annotated
    fig, ax = plt.subplots(figsize=(12, 10))
    for fam in families:
        mask = np.array(family_labels) == fam
        ax.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1],
                  label=fam, s=150, alpha=0.6, color=colors[fam], edgecolors='black', linewidth=1.2)
    
    # Annotate points
    for i, label in enumerate(circuit_labels):
        ax.annotate(label, (umap_embeddings[i, 0], umap_embeddings[i, 1]),
                   fontsize=9, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel("UMAP 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight='bold')
    ax.set_title("Circuit Embeddings (UMAP) - Annotated", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_annotated.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'umap_annotated.png'}")
    plt.close()
    
    # Plot 3: Interactive plot (save as HTML for interactivity)
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        for fam in families:
            mask = np.array(family_labels) == fam
            indices = np.where(mask)[0]
            
            hover_text = [circuit_labels[i] for i in indices]
            
            fig.add_trace(go.Scatter(
                x=umap_embeddings[mask, 0],
                y=umap_embeddings[mask, 1],
                mode='markers',
                name=fam,
                text=hover_text,
                hoverinfo='text',
                marker=dict(size=12, opacity=0.7, line=dict(width=1.5, color='black'))
            ))
        
        fig.update_layout(
            title="Circuit Embeddings (UMAP) - Interactive",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=1000,
            height=800,
            hovermode='closest',
            template='plotly_white'
        )
        
        html_path = output_dir / 'umap_interactive.html'
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")
    except ImportError:
        print("Note: plotly not installed. Skipping interactive plot.")


def save_metadata(umap_embeddings: np.ndarray, circuit_labels: List[str],
                 family_labels: List[str], output_dir: str = '.'):
    """Save embeddings and metadata to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metadata = {
        'circuit_labels': circuit_labels,
        'family_labels': family_labels,
        'num_circuits': len(circuit_labels),
        'embedding_dim': umap_embeddings.shape[1],
    }
    
    with open(output_dir / 'umap_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    np.save(output_dir / 'umap_embeddings.npy', umap_embeddings)
    
    print(f"Saved: {output_dir / 'umap_metadata.json'}")
    print(f"Saved: {output_dir / 'umap_embeddings.npy'}")


def main(args):
    device = torch.device('cpu')
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Prepare circuit specs
    circuit_specs = config['data']['circuits']
    if args.include_val:
        test_circuits = config['data'].get('test_circuits', {})
        for family, ids in test_circuits.items():
            if family not in circuit_specs:
                circuit_specs[family] = []
            circuit_specs[family].extend(ids)
    
    # Extract embeddings
    embeddings, circuit_labels, family_labels = extract_embeddings(
        model, None, circuit_specs, config['data']['data_dir'], device
    )
    
    print(f"\nExtracted {len(circuit_labels)} circuit embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Apply UMAP
    umap_embeddings = apply_umap(embeddings, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    
    # Create visualizations
    plot_embeddings(umap_embeddings, circuit_labels, family_labels, args.output_dir)
    
    # Save metadata
    save_metadata(umap_embeddings, circuit_labels, family_labels, args.output_dir)
    
    print(f"\nVisualization complete! Check {args.output_dir}/ for outputs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize learned circuit embeddings using UMAP')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_smoke_test/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./umap_results',
                       help='Directory to save visualizations')
    parser.add_argument('--n_neighbors', type=int, default=5,
                       help='UMAP n_neighbors parameter (controls local vs global structure)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (controls tightness of embedding)')
    parser.add_argument('--include_val', action='store_true',
                       help='Include validation circuits in visualization')
    
    args = parser.parse_args()
    main(args)
