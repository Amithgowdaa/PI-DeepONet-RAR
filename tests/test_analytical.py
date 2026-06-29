"""
Tests for the analytical Kirsch solution.
Verifies the stress concentration factor, symmetry, and far-field behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fem_baseline'))

import numpy as np
from analytical_kirsch import (
    analytical_kirsch_stress,
    analytical_kirsch_stress_polar,
    analytical_kirsch_displacement,
    stress_concentration_factor,
)


def test_scf_equals_three():
    """The theoretical SCF for plate with hole under uniaxial tension is exactly 3.0."""
    scf = stress_concentration_factor(R=0.2, T=1.0)
    assert abs(scf - 3.0) < 1e-10, f"SCF should be 3.0, got {scf}"
    print(f"    SCF = {scf:.10f} [OK]")


def test_traction_free_hole():
    """On the hole surface (r=R), radial stress sigma_rr should be zero."""
    R = 0.2
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.full_like(theta, R)

    sigma_rr, _, _ = analytical_kirsch_stress_polar(r, theta, R=R, T=1.0)

    assert np.allclose(sigma_rr, 0.0, atol=1e-10), \
        f"sigma_rr at r=R should be 0, got max |sigma_rr| = {np.abs(sigma_rr).max()}"
    print(f"    max|sigma_rr(r=R)| = {np.abs(sigma_rr).max():.2e} [OK]")


def test_far_field_stress():
    """Far from the hole, stresses should approach uniaxial tension: sigma_xx -> T."""
    R = 0.2
    T = 1.0
    # Points very far from hole
    x_far = np.full(50, 100.0)
    y_far = np.linspace(-10, 10, 50)

    sxx, syy, sxy = analytical_kirsch_stress(x_far, y_far, R=R, T=T)

    assert np.allclose(sxx, T, atol=1e-3), \
        f"sigma_xx far away should be ~T={T}, got mean {sxx.mean():.4f}"
    assert np.allclose(syy, 0.0, atol=1e-3), \
        f"sigma_yy far away should be ~0, got mean {syy.mean():.4f}"
    print(f"    Far-field: sigma_xx={sxx.mean():.6f}, sigma_yy={syy.mean():.6f} [OK]")


def test_symmetry():
    """Stress field should have the correct symmetry properties."""
    R = 0.2
    T = 1.0

    # Points symmetric about x-axis
    x = np.array([0.5, 0.5])
    y = np.array([0.3, -0.3])

    sxx1, syy1, sxy1 = analytical_kirsch_stress(x, y, R=R, T=T)

    # sigma_xx and sigma_yy should be same for y and -y
    assert np.isclose(sxx1[0], sxx1[1], atol=1e-10), \
        f"sigma_xx should be symmetric: {sxx1[0]} vs {sxx1[1]}"
    # sigma_xy should be antisymmetric about x-axis
    assert np.isclose(sxy1[0], -sxy1[1], atol=1e-10), \
        f"sigma_xy should be antisymmetric: {sxy1[0]} vs {sxy1[1]}"
    print("    Symmetry verified [OK]")


def test_displacement_finite():
    """Displacements should be finite everywhere outside the hole."""
    R = 0.2
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    # Keep only points outside the hole
    mask = xx**2 + yy**2 >= R**2
    xx, yy = xx[mask], yy[mask]

    ux, uy = analytical_kirsch_displacement(xx, yy, R=R, T=1.0, E=1.0, nu=0.3)

    assert np.all(np.isfinite(ux)), "u_x has non-finite values"
    assert np.all(np.isfinite(uy)), "u_y has non-finite values"
    print(f"    Displacements finite for {len(xx)} points [OK]")


def test_peak_hoop_stress():
    """
    At (x=0, y=R), the hoop stress sigma_xx should equal 3*T (the SCF location).
    In Cartesian coords at theta=pi/2, sigma_theta maps to sigma_xx.
    """
    R = 0.2
    T = 1.0
    sxx, _, _ = analytical_kirsch_stress(
        np.array([0.0]), np.array([R]), R=R, T=T
    )
    assert np.isclose(sxx[0], 3.0 * T, atol=1e-10), \
        f"sigma_xx at (0, R) should be 3*T={3*T}, got {sxx[0]}"
    print(f"    sigma_xx at (0, R) = {sxx[0]:.6f} = 3T [OK]")
