import numpy as np

class GaussianRandomField:
    """
    1D Gaussian Random Field (GRF) generator.
    Used to sample random boundary/loading functions for the Branch net input.
    """
    def __init__(self, num_sensors=100, length_scale=0.2, variance=1.0, domain=(0.0, 1.0)):
        self.num_sensors = num_sensors
        self.length_scale = length_scale
        self.variance = variance
        self.domain = domain
        self.sensors = np.linspace(domain[0], domain[1], num_sensors)
        
        # Build covariance matrix
        x1, x2 = np.meshgrid(self.sensors, self.sensors)
        dist = np.abs(x1 - x2)
        self.K = variance * np.exp(-0.5 * (dist / length_scale)**2)
        
        # Jitter for numerical stability
        self.K += 1e-8 * np.eye(num_sensors)
        self.L = np.linalg.cholesky(self.K)

    def sample(self, num_samples):
        """
        Samples random loading functions at the sensor locations.
        
        Args:
            num_samples: Number of loading functions to sample.
        Returns:
            samples: Array of shape (num_samples, num_sensors).
        """
        u = np.random.normal(size=(self.num_sensors, num_samples))
        samples = np.dot(self.L, u).T
        return samples

if __name__ == "__main__":
    # Test sample generation
    grf = GaussianRandomField(num_sensors=100, length_scale=0.2)
    samples = grf.sample(5)
    print(f"Generated {samples.shape[0]} GRF samples with {samples.shape[1]} sensors.")
