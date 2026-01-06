"""Smoke test: import modules and run a single forward + loss step."""
import sys
import numpy as np
import torch

from gnn_training.models import ContrastiveGINModel
from gnn_training.losses import NTXentLoss


def make_tiny_graph(num_nodes=4, input_dim=10):
    features = np.random.rand(num_nodes, input_dim).astype(np.float32)
    adjacency = np.eye(num_nodes, dtype=np.float32)
    graph = {
        'features': features,
        'adjacency': adjacency,
        'nodes': [f'n{i}' for i in range(num_nodes)]
    }
    return graph


def main():
    try:
        device = torch.device('cpu')

        # Create two augmented views (for smoke test we use identical graphs)
        g1 = make_tiny_graph()
        g2 = make_tiny_graph()

        # Build model
        model = ContrastiveGINModel(
            input_dim=10,
            hidden_dims=[16, 16],
            embedding_dim=32,
            projection_dim=16,
            dropout=0.0,
            use_batch_norm=False,
        ).to(device)

        # Encode graphs
        h1 = torch.tensor(g1['features'], dtype=torch.float32, device=device)
        adj1 = torch.tensor(g1['adjacency'], dtype=torch.float32, device=device)
        h2 = torch.tensor(g2['features'], dtype=torch.float32, device=device)
        adj2 = torch.tensor(g2['adjacency'], dtype=torch.float32, device=device)

        z1 = model.encode(h1, adj1).unsqueeze(0)  # [1, emb]
        z2 = model.encode(h2, adj2).unsqueeze(0)

        # Projection outputs
        p1 = model(h1, adj1).unsqueeze(0)
        p2 = model(h2, adj2).unsqueeze(0)

        # Loss
        loss_fn = NTXentLoss(temperature=0.07)
        loss = loss_fn(z1, z2)

        print("Smoke test successful.")
        print(f"Encoder embedding shape: {z1.shape}")
        print(f"Projection shape: {p1.shape}")
        print(f"NT-Xent loss: {loss.item():.6f}")

    except Exception as e:
        print("Smoke test failed:", e)
        raise


if __name__ == '__main__':
    main()
