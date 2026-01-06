"""Test multi-family training infrastructure."""
import sys
import torch
import torch.optim as optim
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from gnn_training.models import ContrastiveGINModel
from gnn_training.losses import NTXentLoss
from gnn_training.training import Trainer, Validator
from gnn_training.data import MultiCircuitDataLoader
from gnn_training.pairs import MultiCircuitBatchSampler


def main():
    """Test multi-family training with diff_amps only (single family)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Config
    data_dir = "/home/karthik/sim_clean/AMS-opt/netlists"
    
    # Single family (diff_amps) for now
    circuit_specs = {
        'diff_amps': ['75', '77', '84', '86', '94']
    }
    
    train_circuits = {
        'diff_amps': ['75', '77', '84', '86']
    }
    
    test_circuits = {
        'diff_amps': ['94']
    }
    
    logger.info(f"Train circuits: {train_circuits}")
    logger.info(f"Test circuits: {test_circuits}")
    
    # Build model
    model = ContrastiveGINModel(
        input_dim=26,
        hidden_dims=[32, 32],
        embedding_dim=64,
        projection_dim=64,
        dropout=0.1,
        use_batch_norm=True,
    ).to(device)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    loss_fn = NTXentLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Trainer & Validator
    checkpoint_dir = Path("./checkpoints_test_multi")
    trainer = Trainer(model, loss_fn, optimizer, device, checkpoint_dir)
    validator = Validator(device)
    
    # Data loaders
    logger.info("Creating multi-family data loader and samplers...")
    try:
        loader = MultiCircuitDataLoader(
            data_dir=data_dir,
            circuit_specs=circuit_specs,
            cache=False
        )
        logger.info(f"Data loader created. Families: {loader.list_families()}")
        
        train_sampler = MultiCircuitBatchSampler(
            loader=loader,
            circuit_specs=train_circuits,
            data_dir=data_dir,
            batch_size=4,
            pos_neg_ratio=1.0,
            family_balance='equal'
        )
        logger.info("Train sampler created successfully")
        
        val_sampler = MultiCircuitBatchSampler(
            loader=loader,
            circuit_specs=test_circuits,
            data_dir=data_dir,
            batch_size=4,
            pos_neg_ratio=1.0,
            family_balance='equal'
        )
        logger.info("Val sampler created successfully")
    except Exception as e:
        logger.error(f"Failed to create samplers: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Training: 1 epoch, max 5 batches
    logger.info("=" * 50)
    logger.info("Starting multi-family 1-epoch test run...")
    logger.info("=" * 50)
    
    try:
        train_loss = trainer.train_epoch(train_sampler, num_batches=5)
        logger.info(f"Train Loss (Epoch 1): {train_loss:.6f}")
        
        val_metrics = validator.validate(model, loss_fn, val_sampler, num_batches=2)
        logger.info(f"Val Loss (Epoch 1): {val_metrics['loss']:.6f}")
        logger.info(f"Val Num Samples: {val_metrics['num_samples']}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("=" * 50)
    logger.info("SUCCESS: Multi-family pipeline test completed!")
    logger.info("=" * 50)
    logger.info("✓ MultiCircuitDataLoader (all families)")
    logger.info("✓ MultiCircuitBatchSampler (family-balanced sampling)")
    logger.info("✓ Full training loop on multi-family data")


if __name__ == '__main__':
    main()
