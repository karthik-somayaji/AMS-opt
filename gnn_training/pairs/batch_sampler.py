"""Batch sampling for contrastive learning."""
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Iterator
from .pair_generator import PairGenerator
from .sampling import HardNegativeMiner


class ContrastiveBatchSampler:
    """Create balanced batches with positive and negative pairs."""
    
    def __init__(self, all_circuit_ids: List[str], data_dir: str, circuit_family_dir: str = "diff_amps",
                 batch_size: int = 32, pos_neg_ratio: float = 1.0):
        """
        Args:
            all_circuit_ids: list of circuit IDs in this family
            data_dir: base path
            circuit_family_dir: e.g., "diff_amps"
            batch_size: total samples per batch
            pos_neg_ratio: ratio of positive to negative samples (e.g., 1.0 = 50% pos, 50% neg)
        """
        self.all_circuits = all_circuit_ids
        self.data_dir = data_dir
        self.circuit_family_dir = circuit_family_dir
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        
        # Initialize pair generators and hard negative miner
        self.pair_generators = {
            cid: PairGenerator(cid, data_dir, circuit_family_dir)
            for cid in all_circuit_ids
        }
        self.hard_neg_miner = HardNegativeMiner(all_circuit_ids, data_dir, circuit_family_dir)
        
        # Compute batch composition
        self._compute_batch_sizes()
    
    def _compute_batch_sizes(self):
        """Compute how many positive and negative samples per batch."""
        total = self.batch_size
        # Assume pairs contribute 2 samples each (anchor + positive/negative)
        # So if we have pos_neg_ratio = 1.0, we want 50% of samples to be positive pairs
        
        # Total pairs = batch_size / 2 (since each pair is 2 samples)
        total_pairs = total // 2
        
        # num_pos_pairs / (num_pos_pairs + num_neg_pairs) = pos_neg_ratio
        self.num_pos_pairs = int(total_pairs * self.pos_neg_ratio / (1 + self.pos_neg_ratio))
        self.num_neg_pairs = total_pairs - self.num_pos_pairs
    
    def __iter__(self) -> Iterator[Dict[str, List[Any]]]:
        """
        Yield batches of samples.
        
        Yields:
            Dict with keys:
                'anchors': list of anchor graphs
                'positives': list of positive graphs (same length as anchors)
                'hard_negatives': list of hard negative graphs
                'neg_circuit_ids': corresponding circuit IDs for negatives
        """
        while True:
            batch = self._sample_batch()
            yield batch
    
    def _sample_batch(self) -> Dict[str, List[Any]]:
        """Sample a single batch."""
        batch = {
            'anchors': [],
            'positives': [],
            'hard_negatives': [],
            'neg_circuit_ids': [],
            'labels': [],  # 1 for positive pair, 0 for negative
        }
        
        # Sample positive pairs
        for _ in range(self.num_pos_pairs):
            anchor_circuit = random.choice(self.all_circuits)
            pair_gen = self.pair_generators[anchor_circuit]
            anchor_graph, pos_graph, pert_type = pair_gen.generate_positive_pair()
            
            batch['anchors'].append(anchor_graph)
            batch['positives'].append(pos_graph)
            batch['labels'].append(1)
        
        # Sample hard negative pairs
        for _ in range(self.num_neg_pairs):
            anchor_circuit = random.choice(self.all_circuits)
            hard_negs = self.hard_neg_miner.get_hard_negatives(anchor_circuit, num_hard=1)
            
            if hard_negs:
                neg_circuit_id, neg_graph, similarity = hard_negs[0]
                batch['anchors'].append(self.pair_generators[anchor_circuit].loader.get_graph())
                batch['hard_negatives'].append(neg_graph)
                batch['neg_circuit_ids'].append(neg_circuit_id)
                batch['labels'].append(0)
            else:
                # Fall back: sample random different circuit
                other_circuits = [c for c in self.all_circuits if c != anchor_circuit]
                if other_circuits:
                    neg_circuit = random.choice(other_circuits)
                    neg_graph = self.hard_neg_miner.loaders[neg_circuit].get_graph()
                    batch['anchors'].append(self.pair_generators[anchor_circuit].loader.get_graph())
                    batch['hard_negatives'].append(neg_graph)
                    batch['neg_circuit_ids'].append(neg_circuit)
                    batch['labels'].append(0)
        
        return batch
