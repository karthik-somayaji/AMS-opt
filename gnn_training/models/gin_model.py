"""GIN (Graph Isomorphism Network) encoder with projection head for contrastive learning."""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List


class GINConv(nn.Module):
    """Single GIN convolution layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, epsilon: float = 0.0):
        """
        Args:
            input_dim: feature dimension of input
            hidden_dim: output feature dimension
            epsilon: learnable parameter for aggregation (default 0.0)
        """
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        
        # MLP for aggregation: (1 + epsilon) * h_v + sum(h_u)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GIN layer.
        
        Args:
            h: node features [num_nodes, input_dim]
            adj: adjacency matrix [num_nodes, num_nodes] (can be sparse or dense)
        
        Returns:
            updated node features [num_nodes, hidden_dim]
        """
        # Aggregate: sum(h_neighbors)
        if isinstance(adj, torch.sparse.FloatTensor):
            agg = torch.sparse.mm(adj, h)  # sparse matrix multiply
        else:
            agg = torch.mm(adj, h)  # dense matrix multiply
        
        # MLP: (1 + epsilon) * h_v + sum(h_u)
        h_updated = self.mlp((1 + self.epsilon) * h + agg)
        
        return h_updated


class GINEncoder(nn.Module):
    """Multi-layer GIN encoder for circuit graphs."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1, use_batch_norm: bool = True):
        """
        Args:
            input_dim: initial node feature dimension (e.g., 10 for circuit nodes)
            hidden_dims: list of hidden dimensions for each GIN layer (e.g., [64, 64, 32])
            output_dim: final embedding dimension (e.g., 128)
            dropout: dropout rate
            use_batch_norm: whether to apply batch norm after each layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Build GIN layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout)
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.gin_layers.append(GINConv(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final MLP readout: hidden_dims[-1] -> output_dim
        self.readout_mlp = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GIN encoder.
        
        Args:
            h: node features [num_nodes, input_dim]
            adj: adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            graph embedding [output_dim]
        """
        # Apply GIN layers with ReLU and dropout
        for i, gin_layer in enumerate(self.gin_layers):
            h = gin_layer(h, adj)
            
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            
            h = torch.relu(h)
            h = self.dropout(h)
        
        # Readout: average pooling over nodes
        graph_repr = h.mean(dim=0)  # [hidden_dims[-1]]
        
        # Final projection to output_dim
        embedding = self.readout_mlp(graph_repr)  # [output_dim]
        
        return embedding


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim: dimension of encoder output (e.g., 128)
            hidden_dim: hidden dimension (e.g., 128)
            output_dim: projection dimension (e.g., 128)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder output for contrastive loss.
        
        Args:
            x: encoder embedding [embedding_dim]
        
        Returns:
            projected embedding [output_dim]
        """
        return self.mlp(x)


class ContrastiveGINModel(nn.Module):
    """Complete model: GIN encoder + projection head."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], embedding_dim: int,
                 projection_dim: int, dropout: float = 0.1, use_batch_norm: bool = True):
        """
        Args:
            input_dim: initial node feature dimension
            hidden_dims: GIN hidden dimensions
            embedding_dim: dimension of GIN encoder output
            projection_dim: dimension of projection head output
            dropout: dropout rate
            use_batch_norm: whether to use batch norm
        """
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dims, embedding_dim, dropout, use_batch_norm)
        self.projection_head = ProjectionHead(embedding_dim, embedding_dim, projection_dim)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor, return_embedding: bool = False) \
            -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: node features [num_nodes, input_dim]
            adj: adjacency matrix [num_nodes, num_nodes]
            return_embedding: if True, return both embedding and projection; else just projection
        
        Returns:
            projection [projection_dim] or (embedding, projection) if return_embedding=True
        """
        embedding = self.encoder(h, adj)
        projection = self.projection_head(embedding)
        
        if return_embedding:
            return embedding, projection
        return projection
    
    def encode(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Get encoder embedding (without projection)."""
        return self.encoder(h, adj)
