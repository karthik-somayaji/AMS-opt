#!/usr/bin/env python3
"""
Create interactive HTML visualization of UMAP embeddings using Plotly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Installing plotly...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.graph_objects as go
    import plotly.express as px


def create_interactive_umap(embeddings_json_path: str, output_path: str):
    """Create interactive UMAP visualization."""
    
    with open(embeddings_json_path) as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings'])
    metadata = data['metadata']
    
    # Create dataframe-like structure
    circuit_ids = [m['circuit_id'] for m in metadata]
    families = [m['family'] for m in metadata]
    norms = [m['embedding_norm'] for m in metadata]
    
    # Color mapping
    family_colors = {
        'diff_amps': '#1f77b4',
        'comparators': '#ff7f0e',
        'ldo': '#2ca02c'
    }
    
    colors = [family_colors.get(f, '#1f77b4') for f in families]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter for each family
    for family in set(families):
        mask = [f == family for f in families]
        emb_family = embeddings[mask]
        cid_family = [circuit_ids[i] for i, m in enumerate(mask) if m]
        norm_family = [norms[i] for i, m in enumerate(mask) if m]
        
        fig.add_trace(go.Scatter(
            x=emb_family[:, 0],
            y=emb_family[:, 1],
            mode='markers+text',
            text=cid_family,
            textposition='top center',
            name=family,
            marker=dict(
                size=12,
                color=family_colors.get(family, '#1f77b4'),
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Family: ' + family + '<br>Norm: %{customdata:.4f}<extra></extra>',
            customdata=norm_family
        ))
    
    fig.update_layout(
        title='<b>GNN Circuit Embeddings - UMAP Projection</b>',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        hovermode='closest',
        width=1000,
        height=800,
        font=dict(size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.9)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
    )
    
    fig.write_html(output_path)
    print(f"✓ Interactive visualization saved to {output_path}")


if __name__ == '__main__':
    embeddings_path = 'umap_results_full/umap_embeddings.json'
    output_path = 'umap_results_full/umap_interactive.html'
    
    create_interactive_umap(embeddings_path, output_path)
