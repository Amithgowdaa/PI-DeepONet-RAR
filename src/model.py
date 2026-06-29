"""
PI-DeepONet Architecture
=========================

Physics-Informed Deep Operator Network for 2D solid mechanics.

Architecture:
    Branch net: Processes discretized load function T(y) at sensor locations.
    Trunk net:  Processes spatial coordinates (x, y).
    Merge:      Dot product to produce displacement fields u(x,y) and v(x,y).

Supports two forward modes:
    - Paired mode:    branch_in.shape[0] == trunk_in.shape[0]
    - Cartesian mode: outputs shape (N_loads, N_coords, 2)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Standard multi-layer perceptron (MLP) for branch and trunk networks."""

    def __init__(self, layers, activation=nn.Tanh()):
        super().__init__()
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(activation)
        self.net = nn.Sequential(*net_layers)

        # Xavier (Glorot) initialization for better convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PIDeepONet(nn.Module):
    """
    Physics-Informed DeepONet architecture.

    The branch net processes the discretized load function and produces
    an embedding of size P * num_outputs. The trunk net processes
    coordinates (x, y) and produces an embedding of size P. The final
    output for each field (u, v) is the dot product of the corresponding
    branch slice with the trunk embedding.

    Args:
        branch_layers: List of layer sizes for the branch net.
                       The last element must equal P * num_outputs.
        trunk_layers:  List of layer sizes for the trunk net.
                       The last element is P.
        num_outputs:   Number of output fields (default: 2 for u, v).
    """

    def __init__(self, branch_layers, trunk_layers, num_outputs=2):
        super().__init__()
        self.num_outputs = num_outputs
        self.branch_dim_out = branch_layers[-1]
        self.trunk_dim_out = trunk_layers[-1]

        # branch_dim_out must equal P * num_outputs
        self.P = self.trunk_dim_out
        assert self.branch_dim_out == self.P * num_outputs, (
            f"Branch output dimension ({self.branch_dim_out}) must equal "
            f"Trunk output dimension ({self.P}) * num_outputs ({num_outputs}). "
            f"Expected branch[-1] = {self.P * num_outputs}."
        )

        self.branch_net = FeedForward(branch_layers)
        self.trunk_net = FeedForward(trunk_layers)

    def forward(self, branch_in, trunk_in):
        """
        Forward pass.

        Args:
            branch_in: (N_loads, num_sensors) — discretized load functions.
            trunk_in:  (N_coords, 2)          — spatial coordinates.

        Returns:
            If paired mode (N_loads == N_coords):
                (N, 2) tensor with columns [u, v].
            If Cartesian mode:
                (N_loads, N_coords, 2) tensor.
        """
        # Branch features: (N_loads, P * num_outputs)
        branch_out = self.branch_net(branch_in)
        # Trunk features: (N_coords, P)
        trunk_out = self.trunk_net(trunk_in)

        if branch_out.shape[0] == trunk_out.shape[0]:
            # Paired mode: each load paired with one coordinate
            u = torch.sum(branch_out[:, :self.P] * trunk_out, dim=1, keepdim=True)
            v = torch.sum(branch_out[:, self.P:] * trunk_out, dim=1, keepdim=True)
            return torch.cat([u, v], dim=1)
        else:
            # Cartesian product mode
            # branch_out: (N_loads, 2, P)
            branch_out = branch_out.view(-1, self.num_outputs, self.P)
            # trunk_out: (N_coords, P)
            # output: (N_loads, N_coords, 2)
            outputs = torch.einsum("bip,cp->bci", branch_out, trunk_out)
            return outputs

    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
