"""Contrastive loss functions for training."""
from .nt_xent_loss import NTXentLoss, WeightedNTXentLoss
from .consistency_loss import ConsistencyLoss, ConsistencyLossAlternative

__all__ = [
    'NTXentLoss',
    'WeightedNTXentLoss',
    'ConsistencyLoss',
    'ConsistencyLossAlternative',
]
