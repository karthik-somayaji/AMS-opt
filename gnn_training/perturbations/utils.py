"""Utility functions for graph manipulation."""
import numpy as np
import copy
from typing import List, Tuple, Dict, Any


class GraphUtils:
    """Common utilities for graph manipulation."""
    
    @staticmethod
    def delete_edges(adjacency: np.ndarray, edge_indices: List[int]) -> np.ndarray:
        """
        Delete edges from adjacency matrix by index.
        
        Args:
            adjacency: (N, N) adjacency matrix
            edge_indices: list of edge indices to delete (flattened indices)
        
        Returns:
            Modified adjacency matrix with edges deleted
        """
        adj_copy = adjacency.copy()
        adj_flat = adj_copy.flatten()
        adj_flat[edge_indices] = 0
        return adj_flat.reshape(adjacency.shape)
    
    @staticmethod
    def get_edge_indices(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (row, col) indices of non-zero entries in adjacency.
        
        Returns:
            (row_indices, col_indices)
        """
        return np.nonzero(adjacency)
    
    @staticmethod
    def delete_nodes(graph: Dict[str, Any], node_indices: List[int]) -> Dict[str, Any]:
        """
        Delete nodes from a graph by index.
        
        Args:
            graph: graph dict with 'features', 'adjacency', 'nodes', 'node_to_idx'
            node_indices: list of node indices to delete
        
        Returns:
            New graph dict with nodes and edges deleted
        """
        graph_copy = copy.deepcopy(graph)
        
        N = graph['features'].shape[0]
        keep_indices = np.setdiff1d(np.arange(N), node_indices)
        
        # Reorder
        new_features = graph['features'][keep_indices]
        new_adjacency = graph['adjacency'][np.ix_(keep_indices, keep_indices)]
        new_nodes = graph['nodes'][keep_indices]
        
        # Rebuild mappings
        new_node_to_idx = {node: i for i, node in enumerate(new_nodes)}
        new_idx_to_node = {i: node for i, node in enumerate(new_nodes)}
        
        graph_copy['features'] = new_features
        graph_copy['adjacency'] = new_adjacency
        graph_copy['nodes'] = new_nodes
        graph_copy['node_to_idx'] = new_node_to_idx
        graph_copy['idx_to_node'] = new_idx_to_node
        
        return graph_copy
    
    @staticmethod
    def get_node_indices_by_prefix(graph: Dict[str, Any], prefix: str) -> List[int]:
        """Get indices of nodes matching a prefix (e.g., 'dev:')."""
        indices = []
        for i, node_id in enumerate(graph['nodes']):
            if str(node_id).startswith(prefix):
                indices.append(i)
        return indices
    
    @staticmethod
    def get_edge_indices_between_types(graph: Dict[str, Any], src_prefix: str, tgt_prefix: str) -> List[Tuple[int, int]]:
        """
        Get edges (as index pairs) between nodes of two types.
        
        Args:
            src_prefix: source node prefix (e.g., 'dev:')
            tgt_prefix: target node prefix (e.g., 'W_')
        
        Returns:
            List of (src_idx, tgt_idx) tuples
        """
        edges = []
        adj = graph['adjacency']
        
        src_indices = set(GraphUtils.get_node_indices_by_prefix(graph, src_prefix))
        tgt_indices = set(GraphUtils.get_node_indices_by_prefix(graph, tgt_prefix))
        
        rows, cols = np.nonzero(adj)
        for r, c in zip(rows, cols):
            if r in src_indices and c in tgt_indices:
                edges.append((r, c))
            elif c in src_indices and r in tgt_indices:
                edges.append((c, r))
        
        return edges
    
    @staticmethod
    def count_kg_edges(graph: Dict[str, Any]) -> int:
        """
        Count Knowledge Graph edges (parameter-to-performance and performance-to-performance).
        
        These are edges between 'W_', 'L_', 'R_', 'IB_' (parameter) nodes and 'Gain', 'CMRR', etc. (performance) nodes.
        """
        count = 0
        param_prefixes = ('W_', 'L_', 'R_', 'IB_', 'C_')
        perf_names = ('Gain', 'CMRR', 'UGF', 'Power')
        
        for i, node_i in enumerate(graph['nodes']):
            for j, node_j in enumerate(graph['nodes']):
                if i < j and graph['adjacency'][i, j]:
                    node_i_str = str(node_i)
                    node_j_str = str(node_j)
                    
                    # Check if edge is between parameter and performance
                    is_param_perf = (
                        (any(node_i_str.startswith(p) for p in param_prefixes) and 
                         any(node_j_str.startswith(pn) or node_j_str == pn for pn in perf_names)) or
                        (any(node_j_str.startswith(p) for p in param_prefixes) and 
                         any(node_i_str.startswith(pn) or node_i_str == pn for pn in perf_names))
                    )
                    
                    # Also count performance-to-performance edges
                    is_perf_perf = (
                        (any(node_i_str.startswith(pn) or node_i_str == pn for pn in perf_names) and
                         any(node_j_str.startswith(pn) or node_j_str == pn for pn in perf_names))
                    )
                    
                    if is_param_perf or is_perf_perf:
                        count += 1
        
        return count
    
    @staticmethod
    def get_kg_edge_indices(graph: Dict[str, Any]) -> List[int]:
        """
        Get flattened indices of KG edges.
        
        Returns:
            List of flattened indices (use with adj.flatten())
        """
        kg_edges = []
        param_prefixes = ('W_', 'L_', 'R_', 'IB_', 'C_')
        perf_names = ('Gain', 'CMRR', 'UGF', 'Power')
        
        adj = graph['adjacency']
        N = adj.shape[0]
        
        for i in range(N):
            for j in range(N):
                if adj[i, j]:
                    node_i_str = str(graph['nodes'][i])
                    node_j_str = str(graph['nodes'][j])
                    
                    is_param_perf = (
                        (any(node_i_str.startswith(p) for p in param_prefixes) and 
                         any(node_j_str.startswith(pn) or node_j_str == pn for pn in perf_names)) or
                        (any(node_j_str.startswith(p) for p in param_prefixes) and 
                         any(node_i_str.startswith(pn) or node_i_str == pn for pn in perf_names))
                    )
                    
                    is_perf_perf = (
                        (any(node_i_str.startswith(pn) or node_i_str == pn for pn in perf_names) and
                         any(node_j_str.startswith(pn) or node_j_str == pn for pn in perf_names))
                    )
                    
                    if is_param_perf or is_perf_perf:
                        kg_edges.append(i * N + j)
        
        return kg_edges
