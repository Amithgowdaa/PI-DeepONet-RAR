"""
Data Generation for PI-DeepONet Branch Net Inputs
===================================================

Generates random boundary/loading functions using a 1D Gaussian Random
Field (GRF). These represent the spatially varying traction T(y) applied
on the right boundary (x = L) of the plate.

The GRF is parameterized by a squared-exponential (RBF) kernel:
    K(x1, x2) = σ² * exp(-|x1 - x2|² / (2 * l²))

where:
    σ² = variance (controls the amplitude of load fluctuations)
    l  = length_scale (controls the smoothness / spatial correlation)
"""

import numpy as np


class GaussianRandomField:
    """
    1D Gaussian Random Field (GRF) generator.
    Used to sample random boundary/loading functions for the Branch net input.
    """

    def __init__(self, num_sensors=100, length_scale=0.2, variance=1.0,
                 domain=(-1.0, 1.0)):
        """
        Args:
            num_sensors:  Number of sensor points (= branch net input dim).
            length_scale: Correlation length of the RBF kernel.
            variance:     Marginal variance of the field.
            domain:       Tuple (min, max) for sensor locations.
        """
        self.num_sensors = num_sensors
        self.length_scale = length_scale
        self.variance = variance
        self.domain = domain
        self.sensors = np.linspace(domain[0], domain[1], num_sensors)

        # Build covariance matrix (squared-exponential / RBF kernel)
        x1, x2 = np.meshgrid(self.sensors, self.sensors)
        dist = np.abs(x1 - x2)
        self.K = variance * np.exp(-0.5 * (dist / length_scale) ** 2)

        # Jitter for numerical stability of Cholesky
        self.K += 1e-8 * np.eye(num_sensors)
        self.L = np.linalg.cholesky(self.K)

    def sample(self, num_samples, rng=None):
        """
        Samples random loading functions at the sensor locations.

        Args:
            num_samples: Number of loading functions to sample.
            rng:         Optional numpy random Generator for reproducibility.

        Returns:
            samples: Array of shape (num_samples, num_sensors).
        """
        if rng is None:
            u = np.random.normal(size=(self.num_sensors, num_samples))
        else:
            u = rng.normal(size=(self.num_sensors, num_samples))
        samples = np.dot(self.L, u).T
        return samples

    def sample_with_bias(self, num_samples, high_freq_weight=0.5, rng=None):
        """
        Samples load functions biased toward higher-frequency content.

        This is useful for RAR load refinement: the model typically
        struggles with rapidly varying loads, so biasing toward
        high-frequency content helps find challenging load patterns.

        Args:
            num_samples:     Number of loading functions to sample.
            high_freq_weight: Weight for the short-length-scale component
                             (0 = same as standard, 1 = very high frequency).
            rng:             Optional numpy random Generator.

        Returns:
            samples: Array of shape (num_samples, num_sensors).
        """
        # Standard (smooth) samples
        smooth = self.sample(num_samples, rng=rng)

        # Build a short-length-scale GRF for high-frequency component
        short_ls = self.length_scale * 0.3  # 30% of original length scale
        x1, x2 = np.meshgrid(self.sensors, self.sensors)
        dist = np.abs(x1 - x2)
        K_hf = self.variance * np.exp(-0.5 * (dist / short_ls) ** 2)
        K_hf += 1e-8 * np.eye(self.num_sensors)
        L_hf = np.linalg.cholesky(K_hf)

        if rng is None:
            u_hf = np.random.normal(size=(self.num_sensors, num_samples))
        else:
            u_hf = rng.normal(size=(self.num_sensors, num_samples))
        rough = np.dot(L_hf, u_hf).T

        # Blend
        samples = (1.0 - high_freq_weight) * smooth + high_freq_weight * rough
        return samples


if __name__ == "__main__":
    # Test sample generation
    grf = GaussianRandomField(num_sensors=100, length_scale=0.2, domain=(-1.0, 1.0))
    samples = grf.sample(5)
    print(f"Generated {samples.shape[0]} GRF samples with {samples.shape[1]} sensors.")
    print(f"Sample mean: {samples.mean():.4f}, std: {samples.std():.4f}")

    biased = grf.sample_with_bias(5, high_freq_weight=0.7)
    print(f"\nBiased samples: mean={biased.mean():.4f}, std={biased.std():.4f}")
