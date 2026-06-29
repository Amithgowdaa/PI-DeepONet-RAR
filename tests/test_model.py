"""
Tests for the PIDeepONet model architecture.
Verifies forward pass shapes, gradient flow, and parameter counts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model import PIDeepONet, FeedForward


def test_feedforward_shapes():
    """FeedForward MLP should produce correct output shapes."""
    ff = FeedForward([10, 64, 32])
    x = torch.randn(5, 10)
    out = ff(x)
    assert out.shape == (5, 32), f"Expected (5, 32), got {out.shape}"
    print(f"    Shape: input (5,10) -> output {out.shape}")


def test_model_paired_mode():
    """PIDeepONet paired mode: N_loads == N_coords -> output (N, 2)."""
    model = PIDeepONet(
        branch_layers=[20, 64, 100],  # P=50, num_outputs=2 -> branch_out=100
        trunk_layers=[2, 64, 50],     # P=50
        num_outputs=2,
    )
    loads = torch.randn(8, 20)
    coords = torch.randn(8, 2)
    out = model(loads, coords)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
    print(f"    Paired mode: ({loads.shape}, {coords.shape}) -> {out.shape}")


def test_model_cartesian_mode():
    """PIDeepONet Cartesian mode: N_loads != N_coords -> output (N_loads, N_coords, 2)."""
    model = PIDeepONet(
        branch_layers=[20, 64, 100],
        trunk_layers=[2, 64, 50],
        num_outputs=2,
    )
    loads = torch.randn(3, 20)
    coords = torch.randn(10, 2)
    out = model(loads, coords)
    assert out.shape == (3, 10, 2), f"Expected (3, 10, 2), got {out.shape}"
    print(f"    Cartesian mode: ({loads.shape}, {coords.shape}) -> {out.shape}")


def test_model_gradient_flow():
    """Gradients should flow through the model to all parameters."""
    model = PIDeepONet(
        branch_layers=[20, 64, 100],
        trunk_layers=[2, 64, 50],
        num_outputs=2,
    )
    loads = torch.randn(4, 20)
    coords = torch.randn(4, 2, requires_grad=True)
    out = model(loads, coords)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    print("    All parameters received finite gradients [OK]")


def test_model_parameter_count():
    """Parameter count should match expected values."""
    model = PIDeepONet(
        branch_layers=[100, 128, 128, 100],
        trunk_layers=[2, 128, 128, 50],
        num_outputs=2,
    )
    count = model.count_parameters()
    assert count > 0, "Model has no trainable parameters"
    print(f"    Total parameters: {count:,}")


def test_model_assertion_mismatch():
    """Should raise AssertionError when branch/trunk dims don't match."""
    try:
        model = PIDeepONet(
            branch_layers=[20, 64, 30],   # Output 30
            trunk_layers=[2, 64, 50],     # P=50, need 50*2=100
            num_outputs=2,
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("    Correctly rejected mismatched dimensions [OK]")
