import numpy as np

class RemoveKnowledgeNodes:
    def __init__(self, graph):
        """
        Initializes the augmentation class. 
        No parameters are needed for this specific operation.
        """
        self.graph = graph

    def apply(self):
        """
        Removes 'knowledge' nodes from the graph.
        Returns a new graph dictionary (does not operate in-place).
        """
        nodes = self.graph['nodes']
        features = self.graph['features']
        adjacency = self.graph['adjacency']
        
        N = nodes.shape[0]
        
        # Determine if the arrays are flattened based on your (N*D) and (N*N) description
        is_flat_features = (features.ndim == 1)
        is_flat_adj = (adjacency.ndim == 1)
        
        # Reshape to 2D for easier slicing if they are flattened
        if is_flat_features:
            D = features.shape[0] // N
            features_2d = features.reshape((N, D))
        else:
            features_2d = features
            
        if is_flat_adj:
            adjacency_2d = adjacency.reshape((N, N))
        else:
            adjacency_2d = adjacency

        # 1. Create a boolean mask of nodes to KEEP.
        # A knowledge node has ANY of the first 3 features != 0. 
        # Therefore, we keep nodes where ALL of the first 3 features == 0.
        keep_mask = np.all(features_2d[:, :3] == 0, axis=1)
        
        # 2. Filter nodes, features, and adjacency matrix using the mask
        new_nodes = nodes[keep_mask]
        new_features = features_2d[keep_mask]
        
        # For adjacency, we slice both the rows and the columns
        new_adjacency = adjacency_2d[keep_mask][:, keep_mask]
        
        # 3. Flatten back if the original input was flat
        if is_flat_features:
            new_features = new_features.flatten()
        if is_flat_adj:
            new_adjacency = new_adjacency.flatten()
            
        # Return a brand new dictionary
        return {
            'nodes': new_nodes,
            'features': new_features,
            'adjacency': new_adjacency
        }