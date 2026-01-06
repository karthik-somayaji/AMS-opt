"""Multi-circuit and multi-family data loading for contrastive learning."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class MultiCircuitDataLoader:
    """Load graphs from multiple circuit families and individual circuits."""
    
    def __init__(self, data_dir: str, circuit_specs: Dict[str, List[str]], cache: bool = False):
        """
        Args:
            data_dir: base data directory (e.g., netlists/)
            circuit_specs: dict mapping family_name -> list of circuit_ids
                e.g., {'diff_amps': ['75', '77', '84'], 'ldo': ['1', '2'], ...}
            cache: whether to cache graphs in memory
        """
        self.data_dir = Path(data_dir)
        self.circuit_specs = circuit_specs
        self.cache_enabled = cache
        self._cache = {}
        self._graph_cache = {}
        
        # Build circuit -> (family, id) mapping
        self.circuit_to_family = {}
        for family, cids in circuit_specs.items():
            for cid in cids:
                self.circuit_to_family[str(cid)] = family
        
        # Validate all circuits exist
        self._validate_circuits()
        
        # Load metadata (node indices, feature dims, etc.)
        self._load_metadata()
    
    def _validate_circuits(self):
        """Check that all specified circuits have data files."""
        missing = []
        for family, cids in self.circuit_specs.items():
            for cid in cids:
                npz_path = self.data_dir / family / str(cid) / 'comb_graph_gnn.npz'
                if not npz_path.exists():
                    missing.append(f"{family}/{cid}")
        
        if missing:
            raise FileNotFoundError(f"Missing circuit data for: {missing}")
    
    def _load_metadata(self):
        """Load metadata about all circuits."""
        self.metadata = {}
        for family, cids in self.circuit_specs.items():
            self.metadata[family] = {}
            for cid in cids:
                meta_path = self.data_dir / family / str(cid) / 'comb_graph_gnn_meta.json'
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.metadata[family][str(cid)] = json.load(f)
                else:
                    self.metadata[family][str(cid)] = {}
    
    def get_graph(self, circuit_id: str) -> Dict:
        """
        Load a single circuit graph by circuit ID.
        
        Args:
            circuit_id: circuit ID (e.g., '75')
        
        Returns:
            dict with 'nodes', 'features', 'adjacency', 'node_to_idx', 'idx_to_node', 'metadata'
        """
        circuit_id = str(circuit_id)
        
        # Check cache
        if circuit_id in self._graph_cache:
            return self._graph_cache[circuit_id]
        
        # Find family
        if circuit_id not in self.circuit_to_family:
            raise ValueError(f"Circuit {circuit_id} not found in specs")
        
        family = self.circuit_to_family[circuit_id]
        
        # Load NPZ
        npz_path = self.data_dir / family / circuit_id / 'comb_graph_gnn.npz'
        data = np.load(npz_path, allow_pickle=True)
        
        # Build graph dict
        nodes = data['nodes'].tolist() if isinstance(data['nodes'], np.ndarray) else data['nodes']
        features = data['features'].astype(np.float32)
        adjacency = data['adjacency'].astype(np.float32)
        
        # Build index mappings
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n for i, n in enumerate(nodes)}
        
        graph = {
            'nodes': nodes,
            'features': features,
            'adjacency': adjacency,
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'metadata': self.metadata[family].get(circuit_id, {}),
            'circuit_id': circuit_id,
            'family': family,
        }
        
        # Cache if enabled
        if self.cache_enabled:
            self._graph_cache[circuit_id] = graph
        
        return graph
    
    def get_graphs(self, circuit_ids: List[str]) -> Dict[str, Dict]:
        """Load multiple graphs at once."""
        return {cid: self.get_graph(cid) for cid in circuit_ids}
    
    def list_circuits(self, family: Optional[str] = None) -> List[str]:
        """List all circuit IDs, optionally filtered by family."""
        if family is not None:
            return [str(c) for c in self.circuit_specs.get(family, [])]
        else:
            result = []
            for cids in self.circuit_specs.values():
                result.extend([str(c) for c in cids])
            return result
    
    def list_families(self) -> List[str]:
        """List all families."""
        return list(self.circuit_specs.keys())
    
    def get_family(self, circuit_id: str) -> str:
        """Get family name for a circuit."""
        return self.circuit_to_family.get(str(circuit_id))
    
    def clear_cache(self):
        """Clear in-memory cache."""
        self._graph_cache.clear()
