"""Generate positive pairs through perturbations."""
import random
from typing import List, Tuple, Dict, Any
from ..data.graph_loader import CircuitDataLoader
from ..data.prune_parser import PruneParser
from ..perturbations import KGPerturbation, StructuralPerturbation


class PairGenerator:
    """Generate positive pairs for a circuit through perturbations."""
    
    def __init__(self, circuit_id: str, data_dir: str, circuit_family_dir: str = None, 
                 perturbation_types: List[str] = None):
        """
        Args:
            circuit_id: e.g., "75"
            data_dir: base path to circuit families (e.g., netlists/)
            circuit_family_dir: e.g., "diff_amps", "comparators" (used to find prune file)
            perturbation_types: list of perturbation types to use ["kg", "structural"]
        """
        self.circuit_id = circuit_id
        self.data_dir = data_dir
        self.circuit_family_dir = circuit_family_dir or "diff_amps"
        self.perturbation_types = perturbation_types or ["kg", "structural"]
        
        # Load circuit data
        circuit_path = f"{data_dir}/{self.circuit_family_dir}/{circuit_id}"
        self.loader = CircuitDataLoader(circuit_id, f"{data_dir}/{self.circuit_family_dir}")
        
        # Load prune components if structural perturbation is enabled
        self.prune_components = []
        if "structural" in self.perturbation_types:
            # prune_file = f"{circuit_path}/{circuit_id}_prune.txt"
            prune_file = f"{circuit_path}/{circuit_id}_prune.json"
            try:
                prune_parser = PruneParser(prune_file)
                self.prune_components = prune_parser.get_components()
            except Exception as e:
                print(f"Warning: Could not load prune file for {circuit_id}: {e}")
                self.perturbation_types.remove("structural")
    
    def generate_positive_pair(self) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Generate a single positive pair for this circuit.
        
        Returns:
            (anchor_graph, positive_graph, perturbation_type)
        """
        anchor_graph = self.loader.get_graph()
        
        if not self.perturbation_types:
            # No perturbations available, return duplicate
            import copy
            return anchor_graph, copy.deepcopy(anchor_graph), "none"
        
        # Choose random perturbation type
        pert_type = random.choice(self.perturbation_types)
        
        if pert_type == "kg":
            perturb = KGPerturbation(anchor_graph, max_deletion_fraction=0.3)
        elif pert_type == "structural":
            if not self.prune_components:
                # Fall back to KG if no prune components
                perturb = KGPerturbation(anchor_graph, max_deletion_fraction=0.3)
                pert_type = "kg"
            else:
                perturb = StructuralPerturbation(anchor_graph, self.prune_components)
        else:
            raise ValueError(f"Unknown perturbation type: {pert_type}")
        
        positive_graph = perturb.apply()
        
        return anchor_graph, positive_graph, pert_type
    
    def generate_positive_pairs(self, num_pairs: int) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
        """
        Generate multiple positive pairs for this circuit.
        
        Returns:
            List of (anchor, positive, perturbation_type) tuples
        """
        pairs = []
        for _ in range(num_pairs):
            pair = self.generate_positive_pair()
            pairs.append(pair)
        return pairs
