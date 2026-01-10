"""Main training script for contrastive GNN."""
import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import logging
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import from gnn_training modules
from gnn_training.models import ContrastiveGINModel
from gnn_training.losses import NTXentLoss, WeightedNTXentLoss, ConsistencyLoss, ConsistencyLossAlternative
from gnn_training.training import Trainer, Validator, EarlyStopping, LRScheduler
from gnn_training.data import MultiCircuitDataLoader
from gnn_training.pairs import MultiCircuitBatchSampler


def setup_logging(log_dir: Path, verbose: bool = True):
    """Setup logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_dir: Path):
    """Save config to checkpoint directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)


def main(config_path: str = None, **kwargs):
    """Main training function."""
    # Load config
    if config_path is None:
        # Use default config in gnn_training/config/
        config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    
    config = load_config(config_path)
    
    # Override config with command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
    
    logger = setup_logging(Path(config['logging']['tensorboard_dir']), 
                          config['logging']['verbose'])
    
    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Set random seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    
    # Device
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Build model
    model_config = config['model']
    model = ContrastiveGINModel(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        embedding_dim=model_config['embedding_dim'],
        projection_dim=model_config['projection_dim'],
        dropout=model_config.get('dropout', 0.1),
        use_batch_norm=model_config.get('use_batch_norm', True),
    ).to(device)
    
    logger.info(f"Model architecture:\n{model}")
    
    # Build loss function
    loss_config = config['loss']
    if loss_config['type'] == 'nt_xent':
        loss_fn = NTXentLoss(temperature=loss_config.get('temperature', 0.07))
    elif loss_config['type'] == 'weighted_nt_xent':
        loss_fn = WeightedNTXentLoss(
            temperature=loss_config.get('temperature', 0.07),
            hard_neg_weight=loss_config.get('hard_neg_weight', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")
    
    logger.info(f"Loss function: {loss_config['type']}")

    # Optional weighting between NT-Xent and consistency terms
    nt_xent_weight = float(loss_config.get('nt_xent_weight', 1.0))
    logger.info(f"NT-Xent weight: {nt_xent_weight}")
    
    # Build consistency loss if enabled
    consistency_loss_fn = None
    consistency_weight = 0.0
    if loss_config.get('consistency', {}).get('enabled', False):
        consistency_config = loss_config['consistency']
        consistency_type = consistency_config.get('type', 'margin')  # 'margin' or 'mse'
        
        if consistency_type == 'margin':
            consistency_loss_fn = ConsistencyLoss(
                margin=consistency_config.get('margin', 0.5),
                max_dist=consistency_config.get('max_dist', 2.0),
                temperature=consistency_config.get('temperature', 1.0)
            )
        elif consistency_type == 'mse':
            consistency_loss_fn = ConsistencyLossAlternative(
                max_dist=consistency_config.get('max_dist', 2.0)
            )
        else:
            raise ValueError(f"Unknown consistency loss type: {consistency_type}")
        
        consistency_weight = consistency_config.get('weight', 0.1)
        consistency_loss_fn = consistency_loss_fn.to(device)
        logger.info(f"Consistency loss enabled: type={consistency_type}, weight={consistency_weight}")
    
    # Build optimizer
    opt_config = config['optimizer']
    if opt_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0),
            betas=opt_config.get('betas', [0.9, 0.999])
        )
    elif opt_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0),
            momentum=opt_config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    logger.info(f"Optimizer: {opt_config['type']}")
    
    # Build learning rate scheduler
    scheduler = LRScheduler(
        optimizer,
        scheduler_type=config['scheduler'].get('type', 'cosine'),
        total_epochs=config['training']['epochs']
    )
    
    # Build checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    save_config(config, checkpoint_dir)
    
    # Create trainer and validator
    tb_writer = SummaryWriter(config['logging']['tensorboard_dir'])
    trainer = Trainer(model, loss_fn, optimizer, device, checkpoint_dir, tb_writer,
                      consistency_loss_fn=consistency_loss_fn, consistency_weight=consistency_weight,
                      nt_xent_weight=nt_xent_weight)
    validator = Validator(device)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping'].get('patience', 5),
        min_delta=config['early_stopping'].get('min_delta', 0)
    )
    
    # Build data loaders
    data_config = config['data']
    
    # Prepare circuit specs for training and testing
    train_circuits = {}
    test_circuits = {}
    
    for family, circuit_ids in data_config['circuits'].items():
        test_ids = data_config.get('test_circuits', {}).get(family, [])
        train_ids = [c for c in circuit_ids if c not in test_ids]
        
        if train_ids:
            train_circuits[family] = train_ids
        if test_ids:
            test_circuits[family] = test_ids
    
    logger.info(f"Train circuits: {train_circuits}")
    logger.info(f"Test circuits: {test_circuits}")
    
    # Create multi-circuit data loader
    all_circuits = {}
    for family, cids in data_config['circuits'].items():
        all_circuits[family] = [str(c) for c in cids]
    
    loader = MultiCircuitDataLoader(
        data_dir=data_config['data_dir'],
        circuit_specs=all_circuits,
        cache=data_config.get('cache_graphs', False)
    )
    
    # Create batch samplers
    train_sampler = MultiCircuitBatchSampler(
        loader=loader,
        circuit_specs={f: [str(c) for c in cids] for f, cids in train_circuits.items()},
        data_dir=data_config['data_dir'],
        batch_size=data_config['batch_size'],
        pos_neg_ratio=data_config.get('pos_neg_ratio', 1.0),
        family_balance=data_config.get('family_balance', 'equal')
    )
    
    if test_circuits:
        val_sampler = MultiCircuitBatchSampler(
            loader=loader,
            circuit_specs={f: [str(c) for c in cids] for f, cids in test_circuits.items()},
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            pos_neg_ratio=data_config.get('pos_neg_ratio', 1.0),
            family_balance=data_config.get('family_balance', 'equal')
        )
    else:
        val_sampler = None
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = config['training']['epochs']
    log_interval = config['training'].get('log_interval', 10)
    val_interval = config['training'].get('val_interval', 5)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_nt_xent, train_consistency = trainer.train_epoch(train_sampler, num_batches=100)
        
        if (epoch + 1) % log_interval == 0:
            if consistency_weight > 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} (NT-Xent: {train_nt_xent:.4f}, Consistency: {train_consistency:.4f})")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            tb_writer.add_scalar('loss/train_epoch', train_loss, epoch)
            tb_writer.add_scalar('loss/train_nt_xent_epoch', train_nt_xent, epoch)
            if consistency_weight > 0:
                tb_writer.add_scalar('loss/train_consistency_epoch', train_consistency, epoch)
        
        # Validate
        if val_sampler is not None and (epoch + 1) % val_interval == 0:
            val_metrics = validator.validate(model, loss_fn, val_sampler, num_batches=50)
            val_loss = val_metrics['loss']
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
            tb_writer.add_scalar('loss/val_epoch', val_loss, epoch)
            
            # Early stopping
            if early_stopping.should_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch + 1, {'loss': train_loss})
        
        # Step learning rate scheduler
        scheduler.step()
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Current LR: {scheduler.get_lr():.6f}")
    
    logger.info("Training complete!")
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train contrastive GNN for circuit embeddings')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data.circuit_ids', type=int, nargs='+', help='Circuit IDs to train on')
    parser.add_argument('--data.test_circuits', type=int, nargs='+', help='Circuit IDs for testing')
    parser.add_argument('--training.epochs', type=int, help='Number of epochs')
    parser.add_argument('--training.device', type=str, help='Device (cuda/cpu)')
    parser.add_argument('--model.embedding_dim', type=int, help='Embedding dimension')
    
    args = parser.parse_args()
    
    # Convert argparse namespace to kwargs
    kwargs = {}
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            kwargs[key] = value
    
    main(args.config, **kwargs)
