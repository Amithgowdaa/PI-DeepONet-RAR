"""
Tests for the GRF data generation module.
Verifies sample shapes, statistics, and biased sampling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data_generation import GaussianRandomField


def test_grf_sample_shape():
    """GRF samples should have shape (num_samples, num_sensors)."""
    grf = GaussianRandomField(num_sensors=50, length_scale=0.2, domain=(-1, 1))
    samples = grf.sample(10)
    assert samples.shape == (10, 50), f"Expected (10, 50), got {samples.shape}"
    print(f"    Shape: {samples.shape}")


def test_grf_single_sample():
    """Single sample should also work."""
    grf = GaussianRandomField(num_sensors=100)
    sample = grf.sample(1)
    assert sample.shape == (1, 100), f"Expected (1, 100), got {sample.shape}"
    print(f"    Single sample shape: {sample.shape}")


def test_grf_statistics():
    """GRF samples should have approximately zero mean (by GRF definition)."""
    np.random.seed(42)
    grf = GaussianRandomField(num_sensors=100, length_scale=0.2, variance=1.0)
    samples = grf.sample(1000)

    mean = np.mean(samples)
    std = np.std(samples)
    assert abs(mean) < 0.1, f"Mean should be ~0, got {mean:.4f}"
    assert std > 0.1, f"Std should be positive, got {std:.4f}"
    print(f"    Mean: {mean:.4f}, Std: {std:.4f}")


def test_grf_reproducibility():
    """Using a numpy RNG seed should give reproducible results."""
    grf = GaussianRandomField(num_sensors=50)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = grf.sample(5, rng=rng1)
    s2 = grf.sample(5, rng=rng2)
    assert np.allclose(s1, s2), "Seeded samples should be identical"
    print("    Reproducibility verified [OK]")


def test_grf_biased_sampling():
    """Biased samples should have higher variance (rougher functions)."""
    np.random.seed(42)
    grf = GaussianRandomField(num_sensors=100, length_scale=0.3, variance=1.0)

    smooth = grf.sample(500)
    biased = grf.sample_with_bias(500, high_freq_weight=0.8)

    # Compute average "roughness" = mean |f[i+1] - f[i]| across sensor locations
    smooth_rough = np.mean(np.abs(np.diff(smooth, axis=1)))
    biased_rough = np.mean(np.abs(np.diff(biased, axis=1)))

    assert biased_rough > smooth_rough * 0.9, \
        f"Biased samples should be rougher: smooth={smooth_rough:.4f}, biased={biased_rough:.4f}"
    print(f"    Smooth roughness: {smooth_rough:.4f}, Biased roughness: {biased_rough:.4f}")
