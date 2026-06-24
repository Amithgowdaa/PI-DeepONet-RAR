import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Standard multi-layer perceptron (MLP) for branch and trunk networks."""
    def __init__(self, layers, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"act_{i}", activation)
                
    def forward(self, x):
        return self.net(x)

class PIDeepONet(nn.Module):
    """
    Physics-Informed DeepONet architecture.
    - Branch net: Processes the discretized load function (input dimension: number of sensors).
    - Trunk net: Processes the spatial domain coordinates (input dimension: 2 for 2D space).
    - Merges branch and trunk outputs via dot product to output displacements (u, v).
    """
    def __init__(self, branch_layers, trunk_layers, num_outputs=2):
        super().__init__()
        self.num_outputs = num_outputs
        self.branch_dim_out = branch_layers[-1]
        self.trunk_dim_out = trunk_layers[-1]
        
        # We need branch_dim_out to match trunk_dim_out for dot product, 
        # or we split the final outputs to represent multiple output fields (e.g., u and v).
        # Here we assume branch_layers[-1] is P * num_outputs and trunk_layers[-1] is P.
        # So for u: dot(branch[0:P], trunk[0:P]), for v: dot(branch[P:2P], trunk[0:P])
        self.P = self.trunk_dim_out
        assert self.branch_dim_out == self.P * num_outputs, \
            f"Branch output dimension ({self.branch_dim_out}) must equal Trunk output dimension ({self.P}) * num_outputs ({num_outputs})"
            
        self.branch_net = FeedForward(branch_layers)
        self.trunk_net = FeedForward(trunk_layers)
        
    def forward(self, branch_in, trunk_in):
        """
        Args:
            branch_in: (batch_size_loads, num_sensors)
            trunk_in: (batch_size_coords, 2)
        Returns:
            outputs: Tuple of (u, v) each with shape (batch_size_loads, batch_size_coords)
                     for Cartesian product, or (batch_size, 2) if paired.
        """
        # For physics-informed training, we often use paired mode or Cartesian product mode.
        # Let's support both or paired mode by default.
        # Branch features: (batch_size_loads, P * num_outputs)
        branch_out = self.branch_net(branch_in)
        # Trunk features: (batch_size_coords, P)
        trunk_out = self.trunk_net(trunk_in)
        
        # If in paired mode: batch_size_loads == batch_size_coords
        if branch_out.shape[0] == trunk_out.shape[0]:
            # u = sum_k (branch[k, 0:P] * trunk[k, 0:P])
            u = torch.sum(branch_out[:, :self.P] * trunk_out, dim=1, keepdim=True)
            # v = sum_k (branch[k, P:2P] * trunk[k, 0:P])
            v = torch.sum(branch_out[:, self.P:] * trunk_out, dim=1, keepdim=True)
            return torch.cat([u, v], dim=1)
        else:
            # Cartesian product mode
            # branch_out: (N_loads, 2P) -> (N_loads, 2, P)
            branch_out = branch_out.view(-1, self.num_outputs, self.P)
            # trunk_out: (N_coords, P)
            # output: (N_loads, N_coords, 2)
            outputs = torch.einsum("bip,cp->bci", branch_out, trunk_out)
            return outputs
