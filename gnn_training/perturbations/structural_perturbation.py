"""Structural perturbation (Type 2)."""
import random
import copy
from typing import Dict, Any, List
from .utils import GraphUtils


class StructuralPerturbation:
    """Delete random structural nodes from a graph based on prune components."""
    
    def __init__(self, graph: Dict[str, Any], prune_components: List[str]):
        """
        Args:
            graph: graph dict with 'features', 'adjacency', 'nodes', etc.
            prune_components: list of device node names to potentially delete (e.g., ['M0', 'M1', 'R0'])
        """
        self.graph = graph
        self.prune_components = prune_components
        
        # Map component names to node indices
        self.component_to_indices = self._build_component_map()
    
    def _build_component_map(self) -> Dict[str, List[int]]:
        """
        Build mapping from component names to their node indices.
        
        E.g., "M0" -> [2, 3, 4] (device node + terminal nodes)
        """
        comp_map = {}
        
        for component in self.prune_components:
            indices = []
            
            # Find the device node (e.g., "dev:M0")
            dev_node_name = f"dev:{component}"
            for i, node_id in enumerate(self.graph['nodes']):
                if str(node_id) == dev_node_name:
                    indices.append(i)
                    
                    # Also find associated terminal nodes (e.g., "term:M0:D", "term:M0:G", "term:M0:S")
                    for j, node_j in enumerate(self.graph['nodes']):
                        if str(node_j).startswith(f"term:{component}:"):
                            indices.append(j)
            
            if indices:
                comp_map[component] = indices
        
        return comp_map
    
    def apply(self) -> Dict[str, Any]:
        """
        Apply structural perturbation: delete random prune components and their incident edges.
        
        Returns:
            Perturbed graph dict
        """
        if not self.component_to_indices:
            return copy.deepcopy(self.graph)
        
        # Choose random subset of components to delete
        num_to_delete = random.randint(1, len(self.component_to_indices))
        components_to_delete = random.sample(list(self.component_to_indices.keys()), num_to_delete)
        
        # Collect all node indices to delete
        node_indices_to_delete = []
        for component in components_to_delete:
            node_indices_to_delete.extend(self.component_to_indices[component])
        
        # Remove duplicates and sort
        node_indices_to_delete = sorted(set(node_indices_to_delete))
        
        # Delete nodes (and incident edges)
        perturbed = GraphUtils.delete_nodes(self.graph, node_indices_to_delete)
        
        return perturbed
