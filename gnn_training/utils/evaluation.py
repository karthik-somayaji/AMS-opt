"""Utility functions for contrastive learning."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import UMAP
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def compute_similarity_matrix(embeddings: torch.Tensor, similarity_type: str = 'cosine') -> np.ndarray:
    """
    Compute similarity matrix between embeddings.
    
    Args:
        embeddings: [num_samples, embedding_dim]
        similarity_type: 'cosine', 'euclidean', 'dot'
    
    Returns:
        similarity matrix [num_samples, num_samples]
    """
    embeddings = embeddings.detach().cpu().numpy()
    
    if similarity_type == 'cosine':
        return cosine_similarity(embeddings)
    elif similarity_type == 'euclidean':
        from scipy.spatial.distance import pdist, squareform
        return -squareform(pdist(embeddings, metric='euclidean'))
    elif similarity_type == 'dot':
        return np.dot(embeddings, embeddings.T)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")


def get_circuit_embeddings(model: torch.nn.Module, circuit_graphs: Dict[str, Dict],
                          device: torch.device, return_projection: bool = False) -> Dict[str, torch.Tensor]:
    """
    Get embeddings for a set of circuits.
    
    Args:
        model: GNN model
        circuit_graphs: dict mapping circuit_id -> graph_dict
        device: torch device
        return_projection: if True, return projection head outputs; else encoder outputs
    
    Returns:
        dict mapping circuit_id -> embedding [embedding_dim]
    """
    model.eval()
    embeddings = {}
    
    with torch.no_grad():
        for circuit_id, graph in circuit_graphs.items():
            h = torch.tensor(graph['features'], dtype=torch.float32, device=device)
            adj = torch.tensor(graph['adjacency'], dtype=torch.float32, device=device)
            
            if return_projection:
                embedding = model(h, adj, return_embedding=False)
            else:
                embedding = model.encode(h, adj)
            
            embeddings[circuit_id] = embedding.cpu()
    
    return embeddings


def visualize_embeddings_2d(embeddings: Dict[str, torch.Tensor], method: str = 'umap',
                           save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize embeddings in 2D space.
    
    Args:
        embeddings: dict mapping circuit_id -> embedding
        method: 'umap' or 'pca'
        save_path: path to save figure
        figsize: figure size
    """
    # Stack embeddings
    embedding_list = []
    circuit_ids = []
    for cid, emb in embeddings.items():
        embedding_list.append(emb.numpy())
        circuit_ids.append(cid)
    
    X = np.array(embedding_list)
    
    # Reduce to 2D
    if method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)
    
    for i, cid in enumerate(circuit_ids):
        plt.annotate(cid, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title('Circuit Embeddings')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def compute_retrieval_metrics(embeddings: Dict[str, torch.Tensor], similarity_type: str = 'cosine',
                             ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
    """
    Compute retrieval metrics (e.g., recall@k, MAP).
    
    Args:
        embeddings: dict mapping circuit_id -> embedding
        similarity_type: 'cosine', 'euclidean', etc.
        ground_truth: dict mapping circuit_id -> list of positive circuit_ids
    
    Returns:
        dict with metrics
    """
    embedding_list = []
    circuit_ids = []
    for cid in sorted(embeddings.keys()):
        embedding_list.append(embeddings[cid].numpy())
        circuit_ids.append(cid)
    
    X = np.array(embedding_list)
    sim_matrix = compute_similarity_matrix(torch.tensor(X, dtype=torch.float32), similarity_type)
    
    metrics = {}
    
    if ground_truth is not None:
        # Compute recall@k
        for k in [1, 5, 10]:
            recalls = []
            for i, cid in enumerate(circuit_ids):
                if cid not in ground_truth:
                    continue
                
                # Get top-k most similar (excluding self)
                sims = sim_matrix[i].copy()
                sims[i] = -np.inf  # exclude self
                top_k_indices = np.argsort(sims)[::-1][:k]
                top_k_circuits = [circuit_ids[idx] for idx in top_k_indices]
                
                # Compute recall
                hits = sum(1 for tc in top_k_circuits if tc in ground_truth[cid])
                num_positives = len(ground_truth[cid])
                if num_positives > 0:
                    recalls.append(hits / num_positives)
            
            if recalls:
                metrics[f'recall@{k}'] = np.mean(recalls)
    
    return metrics


def normalize_embeddings(embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize embeddings to unit norm."""
    normalized = {}
    for cid, emb in embeddings.items():
        normalized[cid] = F.normalize(emb, p=2, dim=0)
    return normalized
