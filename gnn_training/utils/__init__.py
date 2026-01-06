"""Utility functions for training and evaluation."""
from .evaluation import (
    compute_similarity_matrix,
    get_circuit_embeddings,
    visualize_embeddings_2d,
    compute_retrieval_metrics,
    normalize_embeddings,
)

__all__ = [
    'compute_similarity_matrix',
    'get_circuit_embeddings',
    'visualize_embeddings_2d',
    'compute_retrieval_metrics',
    'normalize_embeddings',
]
