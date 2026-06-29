"""
Tests for the PDE residual computation (2D linear elasticity).
Verifies that known displacement fields produce correct residuals.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from physics import compute_pde_residuals


def test_zero_displacement_zero_residual():
    """
    Zero displacement -> zero stress -> zero PDE residual.
    
    We use u = 0.0 * sin(x), v = 0.0 * sin(y) to maintain a computational
    graph through coords (even though the output is zero). A pure zero
    tensor would lose the grad_fn chain needed for second-order derivatives.
    """
    coords = torch.randn(50, 2, requires_grad=True)

    # Multiply by sin to create a differentiable path, then scale to zero.
    # The key: the graph exists so autograd can compute d(sigma)/d(coords) = 0.
    u_dep = 0.0 * torch.sin(coords[:, 0:1])
    v_dep = 0.0 * torch.sin(coords[:, 1:2])

    res_x, res_y, stresses = compute_pde_residuals(coords, u_dep, v_dep)

    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-6), \
        f"res_x should be ~0, got max {res_x.abs().max().item()}"
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-6), \
        f"res_y should be ~0, got max {res_y.abs().max().item()}"
    print(f"    Zero displacement: max|res_x|={res_x.abs().max().item():.2e}, "
          f"max|res_y|={res_y.abs().max().item():.2e}")


def test_uniform_tension_zero_residual():
    """
    Uniform uniaxial tension: u = x/E, v = -nu*y/E.
    Stresses are constant (sigma_xx=1, others=0), so equilibrium residual = 0.
    
    The key subtlety: constant stresses have zero gradient w.r.t. coords,
    but autograd needs a differentiable path to compute that zero.
    We add a vanishing nonlinear term (0 * sin(x)^2) to keep the graph alive
    through two levels of differentiation.
    """
    E = 1.0
    nu = 0.3
    coords = torch.randn(100, 2, requires_grad=True)

    # Linear displacement + vanishing nonlinear term for graph connectivity
    u = coords[:, 0:1] / E + 0.0 * torch.sin(coords[:, 0:1])**2
    v = -nu * coords[:, 1:2] / E + 0.0 * torch.sin(coords[:, 1:2])**2

    res_x, res_y, stresses = compute_pde_residuals(coords, u, v, E=E, nu=nu)

    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-5), \
        f"res_x should be ~0 for uniform tension, got max {res_x.abs().max().item()}"
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-5), \
        f"res_y should be ~0 for uniform tension, got max {res_y.abs().max().item()}"

    # Check stresses: sigma_xx should be ~1.0
    sxx = stresses['sigma_xx']
    assert torch.allclose(sxx, torch.ones_like(sxx), atol=1e-5), \
        f"sigma_xx should be ~1.0, got mean {sxx.mean().item():.4f}"
    print(f"    Uniform tension: sigma_xx={sxx.mean().item():.4f}, "
          f"max|res|={max(res_x.abs().max().item(), res_y.abs().max().item()):.2e}")


def test_residual_shapes():
    """Output shapes should match input batch size."""
    N = 37
    coords = torch.randn(N, 2, requires_grad=True)
    # Make u,v nonlinear functions of coords so autograd works at all levels
    u = torch.sin(coords[:, 0:1]) * 0.1
    v = torch.cos(coords[:, 1:2]) * 0.1

    res_x, res_y, stresses = compute_pde_residuals(coords, u, v)

    assert res_x.shape == (N, 1), f"res_x shape: expected ({N},1), got {res_x.shape}"
    assert res_y.shape == (N, 1), f"res_y shape: expected ({N},1), got {res_y.shape}"
    assert stresses['sigma_xx'].shape == (N, 1)
    assert stresses['sigma_yy'].shape == (N, 1)
    assert stresses['sigma_xy'].shape == (N, 1)
    print(f"    Shapes correct for N={N}")


def test_residual_finite():
    """Residuals should be finite (no NaN or Inf)."""
    coords = torch.randn(50, 2, requires_grad=True)
    u = torch.tanh(coords[:, 0:1]) * 0.1
    v = torch.tanh(coords[:, 1:2]) * 0.1

    res_x, res_y, stresses = compute_pde_residuals(coords, u, v)

    assert torch.isfinite(res_x).all(), "res_x has non-finite values"
    assert torch.isfinite(res_y).all(), "res_y has non-finite values"
    print("    All residuals are finite [OK]")
