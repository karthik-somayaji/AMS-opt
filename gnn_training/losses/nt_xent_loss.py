"""Contrastive loss functions for GNN training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.
    
    Supports both balanced (equal pos/neg) and imbalanced (more pos than neg) scenarios.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Args:
            temperature: scaling parameter for logits (lower = sharper distribution)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor,
                hard_negs_i: Optional[torch.Tensor] = None,
                hard_negs_j: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch.
        
        Args:
            z_i: anchor embeddings [batch_size, embedding_dim]
            z_j: positive embeddings [batch_size, embedding_dim]
            hard_negs_i: hard negative embeddings for z_i [num_hard, embedding_dim] (optional)
            hard_negs_j: hard negative embeddings for z_j [num_hard, embedding_dim] (optional)
        
        Returns:
            scalar loss
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)  # [batch_size, embedding_dim]
        z_j = F.normalize(z_j, dim=1)  # [batch_size, embedding_dim]
        
        # Compute similarity matrix: z_i @ z_j^T
        # sim[a, b] = cos_sim(z_i[a], z_j[b])
        sim_matrix = torch.mm(z_i, z_j.T) / self.temperature  # [batch_size, batch_size]
        
        # Positive labels: diagonal (z_i[k] should match z_j[k])
        pos_mask = torch.eye(batch_size, device=z_i.device, dtype=torch.bool)
        
        # Compute logits and labels for contrastive loss
        # For each anchor z_i[k], we have:
        #   - 1 positive: z_j[k]
        #   - (batch_size - 1) negatives from cross-batch (z_j[l] for l != k)
        #   - num_hard additional hard negatives (if provided)
        
        loss = 0.0
        
        for k in range(batch_size):
            # Get logits for anchor k: similarities to all positives and negatives
            logits_pos = sim_matrix[k, k:k+1]  # [1] - similarity to positive
            logits_neg = torch.cat([sim_matrix[k, :k], sim_matrix[k, k+1:]])  # [batch_size-1]
            
            if hard_negs_i is not None:
                # Add hard negatives
                hard_negs_i_norm = F.normalize(hard_negs_i, dim=1)
                hard_neg_sims = torch.mm(z_i[k:k+1], hard_negs_i_norm.T) / self.temperature  # [1, num_hard]
                logits_neg = torch.cat([logits_neg, hard_neg_sims.squeeze(0)])
            
            # Combine positive and negative logits
            logits = torch.cat([logits_pos, logits_neg])  # [1 + num_neg + num_hard]
            
            # Labels: 0 corresponds to the positive pair
            labels = torch.zeros(1, device=z_i.device, dtype=torch.long)
            
            # Cross-entropy loss
            loss_k = F.cross_entropy(logits.unsqueeze(0), labels, reduction='mean')
            loss += loss_k
        
        loss = loss / batch_size
        
        return loss


class WeightedNTXentLoss(nn.Module):
    """NT-Xent loss with weighted hard negatives."""
    
    def __init__(self, temperature: float = 0.07, hard_neg_weight: float = 1.0):
        """
        Args:
            temperature: scaling parameter
            hard_neg_weight: weight multiplier for hard negative samples (> 1.0 increases loss)
        """
        super().__init__()
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor,
                hard_negs_i: Optional[torch.Tensor] = None,
                hard_neg_similarities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted NT-Xent loss.
        
        Args:
            z_i: anchor embeddings [batch_size, embedding_dim]
            z_j: positive embeddings [batch_size, embedding_dim]
            hard_negs_i: hard negative embeddings [num_hard, embedding_dim]
            hard_neg_similarities: Jaccard similarities of hard negatives [num_hard]
                (used to weight their contribution; higher overlap = higher weight)
        
        Returns:
            scalar loss
        """
        batch_size = z_i.shape[0]
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        sim_matrix = torch.mm(z_i, z_j.T) / self.temperature
        
        loss = 0.0
        
        for k in range(batch_size):
            logits_pos = sim_matrix[k, k:k+1]
            logits_neg = torch.cat([sim_matrix[k, :k], sim_matrix[k, k+1:]])
            
            if hard_negs_i is not None:
                hard_negs_i_norm = F.normalize(hard_negs_i, dim=1)
                hard_neg_sims = torch.mm(z_i[k:k+1], hard_negs_i_norm.T) / self.temperature
                
                # Weight by overlap (higher overlap = more important negative)
                if hard_neg_similarities is not None:
                    weights = 1.0 + self.hard_neg_weight * hard_neg_similarities
                    hard_neg_sims = hard_neg_sims * weights.unsqueeze(0)
                
                logits_neg = torch.cat([logits_neg, hard_neg_sims.squeeze(0)])
            
            logits = torch.cat([logits_pos, logits_neg])
            labels = torch.zeros(1, device=z_i.device, dtype=torch.long)
            
            loss_k = F.cross_entropy(logits.unsqueeze(0), labels)
            loss += loss_k
        
        loss = loss / batch_size
        
        return loss
