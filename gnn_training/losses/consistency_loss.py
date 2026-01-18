"""Consistency loss: enforce embedding distances to reflect circuit similarity."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConsistencyLoss(nn.Module):
    """
    Weighted consistency loss: embeddings of similar circuits should be closer.
    
    For a batch of N circuits with pairwise similarity scores sim[i,j] ∈ [0,1],
    penalize embeddings whose distances don't respect the similarity structure.
    
    Formula (margin-based):
        loss = Σ_{i,j} max(0, margin + ||z_i - z_j|| - sim[i,j] * max_dist)
    
    Intuition:
    - If sim[i,j] is high (0.7), target distance = 0.7 * max_dist (small)
    - If sim[i,j] is low (0.2), target distance = 0.2 * max_dist (large)
    - Penalizes violations of this relationship
    """
    
    def __init__(self, margin: float = 0.5, max_dist: float = 2.0, 
                 temperature: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: margin between expected and actual distance (for hard constraint)
            max_dist: scaling factor for similarity [0,1] → distance [0, max_dist]
            temperature: softness of the margin constraint (lower = sharper)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.max_dist = max_dist
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, embeddings: torch.Tensor, 
                similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            embeddings: [N, embedding_dim] tensor of embeddings for N circuits
            similarity_matrix: [N, N] tensor of pairwise similarity scores ∈ [0, 1]
        
        Returns:
            scalar loss
        """
        N = embeddings.shape[0]
        
        # Compute pairwise L2 distances
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2*z_i·z_j
        sq_dists = torch.cdist(embeddings, embeddings, p=2) ** 2  # [N, N]
        dists = torch.sqrt(sq_dists + 1e-8)  # [N, N], add eps for numerical stability
        
        # Target distance: higher similarity => smaller target distance.
        # If sim=1, target distance is 0; if sim=0, target distance is max_dist.
        target_dists = (1.0 - similarity_matrix) * self.max_dist  # [N, N]
        
        # Compute margin violation: max(0, margin + actual_dist - target_dist)
        violations = torch.relu(self.margin + dists - target_dists)  # [N, N]
        
        # Apply temperature scaling for soft enforcement
        loss_matrix = violations / self.temperature  # [N, N]
        
        # Mask out self-pairs (diagonal) since sim[i,i] = 1.0 and dist[i,i] = 0
        mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
        masked_loss = loss_matrix * mask.float()
        
        # Aggregate
        if self.reduction == 'mean':
            # Average over all non-diagonal pairs
            loss = masked_loss.sum() / mask.sum().float()
        else:
            loss = masked_loss.sum()
        
        return loss


class ConsistencyLossAlternative(nn.Module):
    """
    Alternative consistency loss using MSE distance between normalized distances and similarities.
    
    More continuous penalty: instead of hard margin, directly minimize
        loss = ||normalized_distance - similarity||^2
    """
    
    def __init__(self, max_dist: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            max_dist: scaling factor for distance normalization
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.max_dist = max_dist
        self.reduction = reduction
    
    def forward(self, embeddings: torch.Tensor,
                similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE-based consistency loss.
        
        Args:
            embeddings: [N, embedding_dim]
            similarity_matrix: [N, N] with values in [0, 1]
        
        Returns:
            scalar loss
        """
        N = embeddings.shape[0]
        
        # Compute distances
        dists = torch.cdist(embeddings, embeddings, p=2)  # [N, N]
        
        # Normalize to [0, 1] range
        max_actual_dist = dists.max().detach() + 1e-8
        normalized_dists = dists / max_actual_dist  # [N, N], approximately [0, 1]
        
        # Compute MSE between normalized distance and similarity
        # High similarity → low distance (inverse relationship)
        # We want: normalized_dist ≈ (1 - similarity)
        target_normalized_dists = 1.0 - similarity_matrix  # [N, N]
        
        mse_loss = F.mse_loss(normalized_dists, target_normalized_dists, reduction='none')
        
        # Mask out diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
        masked_loss = mse_loss * mask.float()
        
        if self.reduction == 'mean':
            loss = masked_loss.sum() / mask.sum().float()
        else:
            loss = masked_loss.sum()
        
        return loss
