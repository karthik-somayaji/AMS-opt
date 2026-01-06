"""Negative pair sampling and hard negative mining."""
import numpy as np
from typing import List, Tuple, Dict, Any
from ..data.graph_loader import CircuitDataLoader
from ..data.substructure_matcher import SubstructureMatcher


class NegativePairSampler:
    """Sample negative pairs from non-overlapping circuits."""
    
    def __init__(self, all_circuit_ids: List[str], data_dir: str, circuit_family_dir: str = "diff_amps"):
        """
        Args:
            all_circuit_ids: list of all circuit IDs in a family
            data_dir: base path
            circuit_family_dir: e.g., "diff_amps"
        """
        self.all_circuits = all_circuit_ids
        self.data_dir = data_dir
        self.circuit_family_dir = circuit_family_dir
        
        self.matcher = SubstructureMatcher()
        self.loaders = {}
        self.overlap_matrix = {}
        
        # Load all loaders and compute overlaps
        self._initialize()
    
    def _initialize(self):
        """Load all circuits and precompute overlap matrix."""
        # Load all circuit data
        for cid in self.all_circuits:
            try:
                loader = CircuitDataLoader(cid, f"{self.data_dir}/{self.circuit_family_dir}")
                self.loaders[cid] = loader
            except Exception as e:
                print(f"Warning: Could not load circuit {cid}: {e}")
        
        # Precompute overlap matrix
        cids = list(self.loaders.keys())
        for i, cid1 in enumerate(cids):
            for j, cid2 in enumerate(cids):
                if i <= j:
                    substruct1 = self.loaders[cid1].get_substructures()
                    substruct2 = self.loaders[cid2].get_substructures()
                    overlap = self.matcher.get_overlap_count(substruct1, substruct2)
                    self.overlap_matrix[(cid1, cid2)] = overlap
                    if i != j:
                        self.overlap_matrix[(cid2, cid1)] = overlap
    
    def get_non_overlapping_circuits(self, circuit_id: str) -> List[str]:
        """Get list of circuits with zero overlap with given circuit."""
        non_overlapping = []
        for other_id in self.all_circuits:
            if other_id != circuit_id:
                overlap = self.overlap_matrix.get((circuit_id, other_id), 0)
                if overlap == 0:
                    non_overlapping.append(other_id)
        return non_overlapping
    
    def sample_negative_pairs(self, circuit_id: str, num_negatives: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Sample negative circuit IDs (zero overlap) and their graphs.
        
        Args:
            circuit_id: anchor circuit
            num_negatives: number of negative samples to return
        
        Returns:
            List of (neg_circuit_id, neg_graph) tuples
        """
        non_overlapping = self.get_non_overlapping_circuits(circuit_id)
        
        if not non_overlapping:
            # No zero-overlap circuits, sample randomly
            non_overlapping = [c for c in self.all_circuits if c != circuit_id]
        
        num_negatives = min(num_negatives, len(non_overlapping))
        sampled = np.random.choice(non_overlapping, size=num_negatives, replace=False)
        
        negatives = []
        for neg_id in sampled:
            if neg_id in self.loaders:
                neg_graph = self.loaders[neg_id].get_graph()
                negatives.append((neg_id, neg_graph))
        
        return negatives


class HardNegativeMiner:
    """Mine hard negatives: highest non-zero-overlap pairs within a circuit family."""
    
    def __init__(self, all_circuit_ids: List[str], data_dir: str, circuit_family_dir: str = "diff_amps"):
        """
        Args:
            all_circuit_ids: list of circuit IDs
            data_dir: base path
            circuit_family_dir: family subdirectory
        """
        self.all_circuits = all_circuit_ids
        self.data_dir = data_dir
        self.circuit_family_dir = circuit_family_dir
        
        self.matcher = SubstructureMatcher()
        self.loaders = {}
        self.overlap_scores = {}
        
        self._initialize()
    
    def _initialize(self):
        """Load circuits and compute overlap scores."""
        for cid in self.all_circuits:
            try:
                loader = CircuitDataLoader(cid, f"{self.data_dir}/{self.circuit_family_dir}")
                self.loaders[cid] = loader
            except Exception as e:
                print(f"Warning: Could not load {cid}: {e}")
        
        # Compute pairwise overlap scores (Jaccard similarity)
        cids = list(self.loaders.keys())
        for i, cid1 in enumerate(cids):
            for j, cid2 in enumerate(cids):
                if i < j:
                    substruct1 = self.loaders[cid1].get_substructures()
                    substruct2 = self.loaders[cid2].get_substructures()
                    overlap_ratio = self.matcher.get_overlap_ratio(substruct1, substruct2)
                    
                    # Store only non-zero overlaps
                    if overlap_ratio > 0:
                        self.overlap_scores[(cid1, cid2)] = overlap_ratio
                        self.overlap_scores[(cid2, cid1)] = overlap_ratio
    
    def get_hard_negatives(self, circuit_id: str, num_hard: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Get hardest negative examples (highest non-zero overlap).
        
        Returns:
            List of (neg_circuit_id, neg_graph, similarity_score)
        """
        candidates = []
        for other_id in self.all_circuits:
            if other_id != circuit_id:
                score = self.overlap_scores.get((circuit_id, other_id), 0)
                if score > 0:  # Non-zero overlap = hard negative
                    candidates.append((other_id, score))
        
        # Sort by score descending (highest similarity = hardest)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        hard_negs = candidates[:num_hard]
        
        result = []
        for neg_id, score in hard_negs:
            if neg_id in self.loaders:
                neg_graph = self.loaders[neg_id].get_graph()
                result.append((neg_id, neg_graph, score))
        
        return result
    
    def get_batch_similarity_matrix(self, circuit_ids: List[str]) -> np.ndarray:
        """
        Compute pairwise overlap similarity matrix for a batch of circuits.
        
        Args:
            circuit_ids: list of circuit IDs (as strings)
        
        Returns:
            NxN numpy array where sim[i,j] = Jaccard overlap ratio ∈ [0, 1]
        """
        N = len(circuit_ids)
        sim_matrix = np.zeros((N, N), dtype=np.float32)
        
        for i, cid1 in enumerate(circuit_ids):
            for j, cid2 in enumerate(circuit_ids):
                if i == j:
                    sim_matrix[i, j] = 1.0  # Perfect self-similarity
                else:
                    # Lookup precomputed overlap score
                    score = self.overlap_scores.get((cid1, cid2), 0.0)
                    sim_matrix[i, j] = score
        
        return sim_matrix
