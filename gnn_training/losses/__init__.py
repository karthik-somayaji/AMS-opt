"""Contrastive loss functions for training."""
from .nt_xent_loss import NTXentLoss, WeightedNTXentLoss, PositiveOnlyContrastiveLoss
from .consistency_loss import ConsistencyLoss, ConsistencyLossAlternative

__all__ = [
    'NTXentLoss',
    'WeightedNTXentLoss',
    'PositiveOnlyContrastiveLoss',
    'ConsistencyLoss',
    'ConsistencyLossAlternative',
]
