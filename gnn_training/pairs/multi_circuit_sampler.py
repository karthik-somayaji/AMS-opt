"""Multi-family batch sampling for contrastive learning."""
import random
from typing import List, Dict, Any, Iterator, Optional
from gnn_training.data import MultiCircuitDataLoader
from .pair_generator import PairGenerator
from .sampling import HardNegativeMiner


class MultiCircuitBatchSampler:
    """Create balanced batches across multiple circuit families."""
    
    def __init__(self, loader: MultiCircuitDataLoader, circuit_specs: Dict[str, List[str]],
                 data_dir: str, batch_size: int = 32, pos_neg_ratio: float = 1.0,
                 family_balance: str = 'equal'):
        """
        Args:
            loader: MultiCircuitDataLoader instance
            circuit_specs: dict mapping family -> list of circuit_ids
            data_dir: base data directory
            batch_size: total samples per batch
            pos_neg_ratio: ratio of positive to negative samples
            family_balance: 'equal' (equal circuits per family) or 'proportional' (by num circuits)
        """
        self.loader = loader
        self.circuit_specs = circuit_specs
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.family_balance = family_balance
        
        # Initialize pair generators per circuit
        self.pair_generators = {}
        for family, cids in circuit_specs.items():
            for cid in cids:
                self.pair_generators[str(cid)] = PairGenerator(
                    str(cid), data_dir, family
                )
        
        # Initialize hard negative miners per family
        self.hard_neg_miners = {}
        for family, cids in circuit_specs.items():
            self.hard_neg_miners[family] = HardNegativeMiner(
                [str(c) for c in cids], data_dir, family
            )

        # Map circuit_id -> family for building batch similarity matrices
        self.circuit_to_family = {}
        for family, cids in circuit_specs.items():
            for cid in cids:
                self.circuit_to_family[str(cid)] = family
        
        # Compute batch composition
        self._compute_batch_sizes()
        
        # Build family-to-circuits mapping for sampling
        self.family_circuits = {f: [str(c) for c in cids] for f, cids in circuit_specs.items()}
    
    def _compute_batch_sizes(self):
        """Compute positive and negative pair counts."""
        total = self.batch_size
        total_pairs = total // 2
        self.num_pos_pairs = int(total_pairs * self.pos_neg_ratio / (1 + self.pos_neg_ratio))
        self.num_neg_pairs = total_pairs - self.num_pos_pairs
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield batches indefinitely."""
        while True:
            yield self._sample_batch()
    
    def _sample_batch(self) -> Dict[str, Any]:
        """Sample a balanced batch across families."""
        batch = {
            'anchors': [],
            'positives': [],
            'hard_negatives': [],
            'neg_circuit_ids': [],
            'anchor_circuit_ids': [],  # Track circuit IDs for consistency loss
            'families': [],
            'labels': [],
            'structural_graphs': [],  # Store pure structural graphs (SG) without knowledge nodes
        }
        
        # Determine which families to sample from
        families = list(self.circuit_specs.keys())
        
        # Sample positive pairs (balanced across families)
        for _ in range(self.num_pos_pairs):
            family = random.choice(families)
            circuit_id = random.choice(self.family_circuits[family])
            
            pair_gen = self.pair_generators[circuit_id]
            anchor_graph, pos_graph, pert_type, structural_graph = pair_gen.generate_positive_pair()
            
            batch['anchors'].append(anchor_graph)
            batch['positives'].append(pos_graph)
            batch['structural_graphs'].append(structural_graph) # Pure structural graph (SG) without knowledge nodes
            batch['anchor_circuit_ids'].append(circuit_id)
            batch['families'].append(family)
            batch['labels'].append(1)
        
        # Sample negative pairs (balanced across families)
        for _ in range(self.num_neg_pairs):
            family = random.choice(families)
            anchor_circuit = random.choice(self.family_circuits[family])
            
            # Get hard negatives from same family
            miner = self.hard_neg_miners[family]
            hard_negs = miner.get_hard_negatives(anchor_circuit, num_hard=1)
            
            anchor_graph = self.pair_generators[anchor_circuit].loader.get_graph()
            
            if hard_negs:
                neg_circuit_id, neg_graph, similarity = hard_negs[0]
                batch['anchors'].append(anchor_graph)
                batch['hard_negatives'].append(neg_graph)
                batch['neg_circuit_ids'].append(neg_circuit_id)
                batch['anchor_circuit_ids'].append(anchor_circuit)
                batch['families'].append(family)
                batch['labels'].append(0)
            else:
                # Fallback: random different circuit in family
                other_circuits = [c for c in self.family_circuits[family] if c != anchor_circuit]
                if other_circuits:
                    neg_circuit = random.choice(other_circuits)
                    neg_graph = self.pair_generators[neg_circuit].loader.get_graph()
                    batch['anchors'].append(anchor_graph)
                    batch['hard_negatives'].append(neg_graph)
                    batch['neg_circuit_ids'].append(neg_circuit)
                    batch['anchor_circuit_ids'].append(anchor_circuit)
                    batch['families'].append(family)
                    batch['labels'].append(0)
        
        # Compute similarity matrix for consistency loss
        if batch['anchor_circuit_ids']:
            cids = batch['anchor_circuit_ids']
            n = len(cids)
            similarity_matrix = [[0.0] * n for _ in range(n)]
            for i, cid1 in enumerate(cids):
                for j, cid2 in enumerate(cids):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                        continue
                    fam1 = self.circuit_to_family.get(cid1)
                    fam2 = self.circuit_to_family.get(cid2)
                    if fam1 is None or fam2 is None or fam1 != fam2:
                        similarity_matrix[i][j] = 0.0
                        continue
                    miner = self.hard_neg_miners[fam1]
                    similarity_matrix[i][j] = float(miner.overlap_scores.get((cid1, cid2), 0.0))
            batch['similarity_matrix'] = similarity_matrix
        
        return batch
