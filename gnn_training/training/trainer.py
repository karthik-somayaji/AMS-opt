"""Training infrastructure for contrastive learning."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime


class Trainer:
    """Trainer for contrastive GNN model."""
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                 device: torch.device, checkpoint_dir: Path, summary_writer: Optional[SummaryWriter] = None,
                 consistency_loss_fn: Optional[nn.Module] = None, consistency_weight: float = 0.0,
                 nt_xent_weight: float = 1.0):
        """
        Args:
            model: GNN model (ContrastiveGINModel)
            loss_fn: loss function (NTXentLoss or similar)
            optimizer: optimizer (Adam, SGD, etc.)
            device: torch device
            checkpoint_dir: directory to save checkpoints
            summary_writer: tensorboard writer (optional)
            consistency_loss_fn: optional consistency loss function
            consistency_weight: weight for consistency loss (default 0.0 = disabled)
            nt_xent_weight: weight for NT-Xent component (default 1.0)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.consistency_loss_fn = consistency_loss_fn
        self.consistency_weight = consistency_weight
        self.nt_xent_weight = nt_xent_weight
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writer = summary_writer
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, batch_loader, num_batches: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            batch_loader: iterator yielding batches
            num_batches: max number of batches to process (for debugging)
        
        Returns:
            tuple of (avg_total_loss, avg_nt_xent_loss, avg_consistency_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_nt_xent = 0.0
        total_consistency = 0.0

        num_pairs_processed = 0
        num_batches_processed = 0
        
        for batch_idx, batch in enumerate(batch_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            num_batches_processed += 1
            
            # Extract graphs from batch
            anchors = batch['anchors']
            positives = batch.get('positives', [])
            similarity_matrix = batch.get('similarity_matrix', None)
            anchor_circuit_ids = batch.get('anchor_circuit_ids', [])

            batch_nt_xent_loss = None
            batch_consistency_loss = None

            # 1) NT-Xent: compute over the batch of positive pairs.
            # This requires batch_size > 1 to provide in-batch negatives.
            num_pos_pairs = min(len(anchors), len(positives))
            if num_pos_pairs > 0:
                z_anchor_list = []
                z_pos_list = []
                for anchor_graph, pos_graph in zip(anchors[:num_pos_pairs], positives[:num_pos_pairs]):
                    h_anchor = torch.tensor(anchor_graph['features'], dtype=torch.float32, device=self.device)
                    adj_anchor = torch.tensor(anchor_graph['adjacency'], dtype=torch.float32, device=self.device)
                    z_anchor_list.append(self.model.encode(h_anchor, adj_anchor))

                    h_pos = torch.tensor(pos_graph['features'], dtype=torch.float32, device=self.device)
                    adj_pos = torch.tensor(pos_graph['adjacency'], dtype=torch.float32, device=self.device)
                    z_pos_list.append(self.model.encode(h_pos, adj_pos))

                z_anchor_b = torch.stack(z_anchor_list, dim=0)  # [B, D]
                z_pos_b = torch.stack(z_pos_list, dim=0)  # [B, D]

                self.optimizer.zero_grad(set_to_none=True)
                batch_nt_xent_loss = self.loss_fn(z_anchor_b, z_pos_b)
                (self.nt_xent_weight * batch_nt_xent_loss).backward()
                self.optimizer.step()

                total_nt_xent += float(batch_nt_xent_loss.item())
                num_pairs_processed += int(num_pos_pairs)
                self.global_step += 1

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        'loss/train_nt_xent_step',
                        float(batch_nt_xent_loss.item()),
                        self.global_step,
                    )

            # 2) Consistency: compute once per batch (over anchor embeddings) if enabled.
            if (
                self.consistency_loss_fn is not None
                and self.consistency_weight > 0
                and similarity_matrix is not None
                and len(anchors) > 1
            ):
                batch_anchor_emb_list = []
                for i, anchor_graph in enumerate(anchors):
                    h_anchor = torch.tensor(anchor_graph['features'], dtype=torch.float32, device=self.device)
                    adj_anchor = torch.tensor(anchor_graph['adjacency'], dtype=torch.float32, device=self.device)
                    z_anchor = self.model.encode(h_anchor, adj_anchor)
                    if i < len(anchor_circuit_ids):
                        batch_anchor_emb_list.append(z_anchor)

                if batch_anchor_emb_list:
                    batch_emb = torch.stack(batch_anchor_emb_list, dim=0)
                    sim_matrix = torch.tensor(similarity_matrix, dtype=torch.float32, device=self.device)
                    if batch_emb.shape[0] == sim_matrix.shape[0]:
                        self.optimizer.zero_grad(set_to_none=True)
                        batch_consistency_loss = self.consistency_loss_fn(batch_emb, sim_matrix)
                        (self.consistency_weight * batch_consistency_loss).backward()
                        self.optimizer.step()

                        total_consistency += float(batch_consistency_loss.item())

                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar(
                                'loss/train_consistency_step',
                                float(batch_consistency_loss.item()),
                                self.global_step,
                            )

            # 3) Total loss bookkeeping (for epoch-level reporting)
            batch_total = 0.0
            if batch_nt_xent_loss is not None:
                batch_total += self.nt_xent_weight * float(batch_nt_xent_loss.item())
            if batch_consistency_loss is not None:
                batch_total += self.consistency_weight * float(batch_consistency_loss.item())
            total_loss += batch_total
        
        # Epoch summary:
        # - avg_nt_xent is averaged over the number of processed positive pairs.
        # - avg_loss is averaged over the number of processed batches (matches bookkeeping used above).
        avg_loss = total_loss / max(num_batches_processed, 1)
        avg_nt_xent = total_nt_xent / max(num_batches_processed, 1)
        avg_consistency = total_consistency / max(num_batches_processed, 1)
        
        return avg_loss, avg_nt_xent, avg_consistency
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'metrics': metrics or {},
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, path)
        
        # Keep only best checkpoint
        if metrics and 'loss' in metrics and metrics['loss'] < self.best_loss:
            self.best_loss = metrics['loss']
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        return checkpoint.get('epoch', 0)


class Validator:
    """Validator for contrastive learning."""
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: torch device
        """
        self.device = device
    
    def validate(self, model: nn.Module, loss_fn: nn.Module, batch_loader,
                 num_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            model: GNN model
            loss_fn: loss function
            batch_loader: validation batch loader
            num_batches: max batches to process
        
        Returns:
            dict with 'loss' and other metrics
        """
        model.eval()
        total_loss = 0.0
        num_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                
                anchors = batch['anchors']
                positives = batch.get('positives', [])
                hard_negatives = batch.get('hard_negatives', [])
                
                for i, (anchor_graph, pos_graph) in enumerate(zip(anchors, positives)):
                    h_anchor = torch.tensor(anchor_graph['features'], dtype=torch.float32, device=self.device)
                    adj_anchor = torch.tensor(anchor_graph['adjacency'], dtype=torch.float32, device=self.device)
                    
                    h_pos = torch.tensor(pos_graph['features'], dtype=torch.float32, device=self.device)
                    adj_pos = torch.tensor(pos_graph['adjacency'], dtype=torch.float32, device=self.device)
                    
                    z_anchor = model.encode(h_anchor, adj_anchor).unsqueeze(0)
                    z_pos = model.encode(h_pos, adj_pos).unsqueeze(0)
                    
                    hard_negs_tensor = None
                    if i < len(hard_negatives) and hard_negatives[i] is not None:
                        hard_neg_graph = hard_negatives[i]
                        h_neg = torch.tensor(hard_neg_graph['features'], dtype=torch.float32, device=self.device)
                        adj_neg = torch.tensor(hard_neg_graph['adjacency'], dtype=torch.float32, device=self.device)
                        hard_negs_tensor = model.encode(h_neg, adj_neg).unsqueeze(0)
                    
                    loss = loss_fn(z_anchor, z_pos, hard_negs_i=hard_negs_tensor)
                    total_loss += loss.item()
                    num_processed += 1
        
        avg_loss = total_loss / max(num_processed, 1)
        
        return {
            'loss': avg_loss,
            'num_samples': num_processed,
        }


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: number of epochs with no improvement to wait
            min_delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class LRScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = 'cosine',
                 total_epochs: int = 100):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: 'cosine', 'step', 'exponential'
            total_epochs: total training epochs
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            self.scheduler = None
    
    def step(self):
        """Step the scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
