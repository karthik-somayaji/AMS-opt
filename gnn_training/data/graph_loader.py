"""Load circuit GNN data from .npz files."""
import os
import numpy as np
from typing import Tuple, Dict, Any, Optional


class CircuitDataLoader:
    """Load and cache GNN features for a circuit."""
    
    def __init__(self, circuit_id: str, data_dir: str, cache: bool = True):
        """
        Args:
            circuit_id: e.g., "75", "77"
            data_dir: path to netlists/diff_amps/
            cache: whether to keep data in memory
        """
        self.circuit_id = circuit_id
        self.data_dir = data_dir
        self.cache = cache
        self._data_cache = None
        
        # Try to find circuit directory
        self.circuit_path = os.path.join(data_dir, str(circuit_id))
        if not os.path.exists(self.circuit_path):
            raise FileNotFoundError(f"Circuit path not found: {self.circuit_path}")
        
        # Try to load metadata
        meta_path = os.path.join(self.circuit_path, 'comb_graph_gnn_meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        import json
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
    
    def _load_npz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load comb_graph_gnn.npz and return (nodes, features, adjacency)."""
        npz_path = os.path.join(self.circuit_path, 'comb_graph_gnn.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        nodes = data['nodes']  # array of node IDs
        features = data['features'].astype(np.float32)  # (N, D)
        adjacency = data['adjacency'].astype(np.uint8)  # (N, N)
        
        return nodes, features, adjacency
    
    def get_graph(self) -> Dict[str, Any]:
        """
        Returns a graph dictionary with nodes, features, adjacency, and node-to-index mapping.
        
        Returns:
            Dict with keys:
                'nodes': array of node IDs
                'features': (N, D) feature matrix
                'adjacency': (N, N) adjacency matrix
                'node_to_idx': dict mapping node_id -> index
                'idx_to_node': dict mapping index -> node_id
                'metadata': metadata dict
        """
        if self.cache and self._data_cache is not None:
            return self._data_cache
        
        nodes, features, adjacency = self._load_npz()
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        graph_dict = {
            'nodes': nodes,
            'features': features,
            'adjacency': adjacency,
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'metadata': self.metadata,
        }
        
        if self.cache:
            self._data_cache = graph_dict
        
        return graph_dict
    
    def get_substructures(self) -> list:
        """Get list of substructure types for this circuit."""
        return self.metadata.get('substructure_types', [])
    
    def get_device_nodes(self) -> list:
        """Get list of device node IDs (nodes of type 'device')."""
        graph = self.get_graph()
        device_nodes = []
        for node_id in graph['nodes']:
            if str(node_id).startswith('dev:'):
                device_nodes.append(node_id)
        return device_nodes
