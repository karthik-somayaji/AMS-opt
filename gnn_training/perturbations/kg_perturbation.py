"""KG edge perturbation (Type 1)."""
import random
import numpy as np
import copy
from typing import Dict, Any, List
from .utils import GraphUtils


class KGPerturbation:
    """Delete random KG edges from a graph (up to max_deletion_fraction)."""
    
    def __init__(self, graph: Dict[str, Any], max_deletion_fraction: float = 0.3):
        """
        Args:
            graph: graph dict with 'features', 'adjacency', etc.
            max_deletion_fraction: maximum fraction of KG edges to delete (0.0 to 1.0)
        """
        self.graph = graph
        self.max_deletion_fraction = max(0.0, min(1.0, max_deletion_fraction))
        self.kg_edge_indices = GraphUtils.get_kg_edge_indices(graph)
    
    def apply(self) -> Dict[str, Any]:
        """
        Apply KG perturbation: delete random KG edges.
        
        Returns:
            Perturbed graph dict
        """
        if not self.kg_edge_indices:
            # No KG edges to delete, return copy
            return copy.deepcopy(self.graph)
        
        # Decide how many to delete
        max_deletable = int(len(self.kg_edge_indices) * self.max_deletion_fraction)
        max_deletable = max(1, max_deletable)  # Delete at least 1
        
        num_to_delete = random.randint(1, max_deletable)
        edges_to_delete = random.sample(self.kg_edge_indices, num_to_delete)
        
        # Create perturbed graph
        perturbed = copy.deepcopy(self.graph)
        
        # Delete edges
        adj = perturbed['adjacency'].copy()
        adj_flat = adj.flatten()
        for edge_idx in edges_to_delete:
            adj_flat[edge_idx] = 0
        perturbed['adjacency'] = adj_flat.reshape(adj.shape)
        
        return perturbed
