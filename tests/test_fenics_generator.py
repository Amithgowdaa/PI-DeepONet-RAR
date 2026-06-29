"""
Tests for FEniCS dataset generation helper logic.
Verifies constitutive stress equations and von Mises stress computations.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fem_baseline'))

from fenics_dataset_generator import von_mises, LMBDA, MU, E_MODULUS, POISSON_RATIO, HAS_FENICS


def test_von_mises_stress():
    """Verify von Mises stress formula for known tensor states."""
    # Pure uniaxial tension sigma_xx = 100, others 0
    vm = von_mises(100.0, 0.0, 0.0)
    assert np.isclose(vm, 100.0), f"Expected 100.0, got {vm}"
    
    # Pure shear tau_xy = 50, others 0 -> von mises = sqrt(3)*50 approx 86.6025
    vm_shear = von_mises(0.0, 0.0, 50.0)
    assert np.isclose(vm_shear, np.sqrt(3.0) * 50.0), f"Expected {np.sqrt(3.0)*50.0}, got {vm_shear}"
    print(f"    von Mises stress calculations verified (HAS_FENICS={HAS_FENICS}) [OK]")


def test_elasticity_constants():
    """Verify plane stress Lame parameters calculation."""
    expected_lmbda = (E_MODULUS * POISSON_RATIO) / (1.0 - POISSON_RATIO**2)
    expected_mu = E_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    
    assert np.isclose(LMBDA, expected_lmbda), f"Lambda calculation mismatch"
    assert np.isclose(MU, expected_mu), f"Mu calculation mismatch"
    print("    Constitutive parameters verified [OK]")
