#!/usr/bin/env python3
"""
Smoke test: Train on 80% of diff_amps + comparators, validate on 20%.
Runs 2 epochs to check end-to-end pipeline.
"""
import yaml
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

NETLISTS = Path("netlists")
FAMILIES = ["diff_amps", "comparators"]

def collect_circuits():
    """Collect all valid circuit directories from both families."""
    circuits = []
    for fam in FAMILIES:
        fam_dir = NETLISTS / fam
        if not fam_dir.exists():
            continue
        for circ_dir in sorted(fam_dir.iterdir()):
            if not circ_dir.is_dir():
                continue
            # Check if comb_graph_gnn.npz exists
            npz = circ_dir / "comb_graph_gnn.npz"
            if npz.exists():
                circuits.append({
                    "family": fam,
                    "circuit_id": int(circ_dir.name) if circ_dir.name.isdigit() else circ_dir.name,
                })
    return circuits

def split_train_val(circuits, train_ratio=0.8):
    """Split circuits into train and validation sets."""
    random.shuffle(circuits)
    split_idx = int(len(circuits) * train_ratio)
    return circuits[:split_idx], circuits[split_idx:]

def main():
    print("Collecting circuits...")
    circuits = collect_circuits()
    print(f"Found {len(circuits)} circuits total")
    
    train, val = split_train_val(circuits, train_ratio=0.8)
    print(f"Train: {len(train)}, Val: {len(val)}")
    print("\nTrain set:")
    for c in sorted(train, key=lambda x: (x['family'], x['circuit_id'])):
        print(f"  {c['family']:15s} {c['circuit_id']}")
    print("\nVal set:")
    for c in sorted(val, key=lambda x: (x['family'], x['circuit_id'])):
        print(f"  {c['family']:15s} {c['circuit_id']}")
    
    # Build config for training based on default_config.yaml
    config = {
        'data': {
            'data_dir': str(NETLISTS.resolve()),
            'circuits': {},
            'test_circuits': {},
            'batch_size': 8,
            'pos_neg_ratio': 1.0,
            'family_balance': 'equal',
            'cache_graphs': False,
            'shuffle': True,
        },
        'model': {
            'input_dim': 26,  # Our feature dimension
            'hidden_dims': [64, 64, 32],
            'embedding_dim': 128,
            'projection_dim': 128,
            'dropout': 0.1,
            'use_batch_norm': True,
        },
        'loss': {
            'type': 'nt_xent',
            'temperature': 0.07,
            'hard_neg_weight': 1.0,
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'betas': [0.9, 0.999],
        },
        'scheduler': {
            'type': 'cosine',
        },
        'training': {
            'epochs': 2,
            'log_interval': 1,
            'val_interval': 1,
            'checkpoint_dir': './checkpoints_smoke_test',
            'seed': 42,
            'device': 'cuda',
        },
        'early_stopping': {
            'patience': 5,
            'min_delta': 1e-4,
        },
        'logging': {
            'tensorboard_dir': './logs_smoke_test',
            'log_level': 'INFO',
            'verbose': True,
        },
    }
    
    # Initialize circuits dict for all families
    for fam in FAMILIES:
        config['data']['circuits'][fam] = []
        config['data']['test_circuits'][fam] = []
    
    # Populate train circuits
    for c in train:
        config['data']['circuits'][c["family"]].append(c["circuit_id"])
    
    # Populate test/val circuits
    for c in val:
        config['data']['test_circuits'][c["family"]].append(c["circuit_id"])
    
    # Write config
    config_path = Path("gnn_training/config/smoke_test_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nWrote config to {config_path}")
    
    # Run training
    print("\n" + "="*60)
    print("Running smoke test training...")
    print("="*60)
    
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "gnn_training/scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd()
    )
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
