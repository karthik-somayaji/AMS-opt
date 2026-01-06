"""Minimal 1-epoch test run to validate full contrastive GNN pipeline."""
import sys
import torch
import torch.optim as optim
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from gnn_training.models import ContrastiveGINModel
from gnn_training.losses import NTXentLoss
from gnn_training.training import Trainer, Validator
from gnn_training.pairs import ContrastiveBatchSampler


def main():
    """Run 1 epoch of training on diff_amps."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Config
    data_dir = "/home/karthik/sim_clean/AMS-opt/netlists"
    circuit_family = "diff_amps"
    train_circuits = ['75', '77', '84', '86']
    test_circuits = ['94']
    batch_size = 4
    pos_neg_ratio = 1.0
    
    logger.info(f"Train circuits: {train_circuits}")
    logger.info(f"Test circuits: {test_circuits}")
    
    # Build model
    model = ContrastiveGINModel(
        input_dim=26,  # GNN features are 26-dimensional (6 device + 4 perf + 16 meaning)
        hidden_dims=[32, 32],
        embedding_dim=64,
        projection_dim=64,
        dropout=0.1,
        use_batch_norm=True,
    ).to(device)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss
    loss_fn = NTXentLoss(temperature=0.07)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Trainer & Validator
    checkpoint_dir = Path("./checkpoints_test")
    trainer = Trainer(model, loss_fn, optimizer, device, checkpoint_dir)
    validator = Validator(device)
    
    # Data loaders
    logger.info("Creating batch samplers...")
    try:
        train_sampler = ContrastiveBatchSampler(
            train_circuits,
            data_dir=data_dir,
            circuit_family_dir=circuit_family,
            batch_size=batch_size,
            pos_neg_ratio=pos_neg_ratio,
        )
        logger.info("Train batch sampler created successfully")
    except Exception as e:
        logger.error(f"Failed to create train batch sampler: {e}")
        raise
    
    try:
        val_sampler = ContrastiveBatchSampler(
            test_circuits,
            data_dir=data_dir,
            circuit_family_dir=circuit_family,
            batch_size=batch_size,
            pos_neg_ratio=pos_neg_ratio,
        )
        logger.info("Val batch sampler created successfully")
    except Exception as e:
        logger.error(f"Failed to create val batch sampler: {e}")
        raise
    
    # Training: 1 epoch, max 5 batches
    logger.info("=" * 50)
    logger.info("Starting 1-epoch training run...")
    logger.info("=" * 50)
    
    num_train_batches = 5
    num_val_batches = 2
    
    try:
        train_loss = trainer.train_epoch(train_sampler, num_batches=num_train_batches)
        logger.info(f"Train Loss (Epoch 1): {train_loss:.6f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Validation
    try:
        val_metrics = validator.validate(model, loss_fn, val_sampler, num_batches=num_val_batches)
        logger.info(f"Val Loss (Epoch 1): {val_metrics['loss']:.6f}")
        logger.info(f"Val Num Samples: {val_metrics['num_samples']}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("=" * 50)
    logger.info("SUCCESS: Full pipeline test completed!")
    logger.info("=" * 50)
    logger.info("✓ Data loading (CircuitDataLoader)")
    logger.info("✓ Pair generation (KG and structural perturbations)")
    logger.info("✓ Hard negative mining (overlap-based sampling)")
    logger.info("✓ Model forward pass (GIN encoder + projection)")
    logger.info("✓ NT-Xent loss computation")
    logger.info("✓ Backward pass and optimizer step")


if __name__ == '__main__':
    main()
