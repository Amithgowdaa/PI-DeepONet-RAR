"""
Analytical Kirsch Solution for a Plate with a Circular Hole
============================================================

Exact closed-form solution for an infinite plate with a circular hole
of radius R under uniaxial far-field tension T in the x-direction.

Reference:
    Kirsch, G. (1898). "Die Theorie der Elastizität und die Bedürfnisse
    der Festigkeitslehre." Zeitschrift des Vereines deutscher Ingenieure, 42.

Notes:
    - Uses plane-stress assumption
    - Non-dimensionalized: stresses normalized by T, displacements by T*R/E
    - The stress concentration factor (SCF) is exactly 3.0 at the hole 
      boundary (r=R) at theta = pi/2 (top/bottom of hole)
"""

import numpy as np


def cart2pol(x, y):
    """Convert Cartesian (x, y) to polar (r, theta)."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def pol2cart(r, theta):
    """Convert polar (r, theta) to Cartesian (x, y)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def analytical_kirsch_stress_polar(r, theta, R=0.2, T=1.0):
    """
    Exact Kirsch stresses in POLAR coordinates.

    Args:
        r:     Radial distance from the center (array-like).
        theta: Angle from the x-axis in radians (array-like).
        R:     Hole radius.
        T:     Far-field uniaxial tension (applied in x-direction).

    Returns:
        sigma_rr:     Radial stress component.
        sigma_tt:     Tangential (hoop) stress component.
        sigma_rt:     Shear stress component.
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    # Avoid division by zero at r = 0
    r = np.clip(r, 1e-12, None)

    a = R / r  # ratio a = R/r

    sigma_rr = (T / 2.0) * (1.0 - a**2) + (T / 2.0) * (1.0 - 4.0 * a**2 + 3.0 * a**4) * np.cos(2.0 * theta)
    sigma_tt = (T / 2.0) * (1.0 + a**2) - (T / 2.0) * (1.0 + 3.0 * a**4) * np.cos(2.0 * theta)
    sigma_rt = -(T / 2.0) * (1.0 + 2.0 * a**2 - 3.0 * a**4) * np.sin(2.0 * theta)

    return sigma_rr, sigma_tt, sigma_rt


def analytical_kirsch_stress(x, y, R=0.2, T=1.0):
    """
    Exact Kirsch stresses in CARTESIAN coordinates.

    Args:
        x, y: Cartesian coordinates (array-like).
        R:    Hole radius.
        T:    Far-field uniaxial tension.

    Returns:
        sigma_xx: Normal stress in x-direction.
        sigma_yy: Normal stress in y-direction.
        sigma_xy: Shear stress.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    r, theta = cart2pol(x, y)

    sigma_rr, sigma_tt, sigma_rt = analytical_kirsch_stress_polar(r, theta, R, T)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2 = cos_t**2
    sin2 = sin_t**2
    cs = cos_t * sin_t

    # Tensor rotation: polar → Cartesian
    sigma_xx = sigma_rr * cos2 + sigma_tt * sin2 - 2.0 * sigma_rt * cs
    sigma_yy = sigma_rr * sin2 + sigma_tt * cos2 + 2.0 * sigma_rt * cs
    sigma_xy = (sigma_rr - sigma_tt) * cs + sigma_rt * (cos2 - sin2)

    return sigma_xx, sigma_yy, sigma_xy


def analytical_kirsch_displacement(x, y, R=0.2, T=1.0, E=1.0, nu=0.3):
    """
    Exact Kirsch displacements in Cartesian coordinates (plane stress).

    Args:
        x, y: Cartesian coordinates (array-like).
        R:    Hole radius.
        T:    Far-field uniaxial tension.
        E:    Young's modulus (use 1.0 for non-dimensional).
        nu:   Poisson's ratio.

    Returns:
        u_x: Displacement in x-direction.
        u_y: Displacement in y-direction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    r, theta = cart2pol(x, y)
    r = np.clip(r, 1e-12, None)

    a = R / r
    kappa = (3.0 - nu) / (1.0 + nu)  # Plane stress

    # Radial and tangential displacements
    u_r = (T * r / (4.0 * E)) * (
        (kappa - 1.0)
        + 2.0 * a**2
        + (1.0 + kappa) * a**2 * np.cos(2.0 * theta)
        - 2.0 * a**4 * np.cos(2.0 * theta)
    )

    u_t = -(T * r / (4.0 * E)) * (
        (1.0 + kappa) * a**2
        + 2.0 * a**4
        - (kappa - 1.0)  # far-field contribution (corrected: κ-1, not κ+1)
    ) * np.sin(2.0 * theta)

    # Convert polar displacements to Cartesian
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u_x = u_r * cos_t - u_t * sin_t
    u_y = u_r * sin_t + u_t * cos_t

    return u_x, u_y


def stress_concentration_factor(R=0.2, T=1.0):
    """
    Returns the theoretical stress concentration factor (SCF) at the hole edge.

    For the Kirsch problem under uniaxial tension:
        SCF = sigma_theta_max / T = 3.0

    This occurs at theta = pi/2 (top of hole) and theta = 3*pi/2 (bottom).
    """
    # Evaluate sigma_tt at r = R, theta = pi/2
    _, sigma_tt, _ = analytical_kirsch_stress_polar(
        r=np.array([R]),
        theta=np.array([np.pi / 2.0]),
        R=R,
        T=T,
    )
    return float(sigma_tt / T)


def generate_validation_data(R=0.2, T=1.0, E=1.0, nu=0.3, L=1.0,
                              num_points=5000, seed=42):
    """
    Generate a validation dataset of (x, y, u, v, sxx, syy, sxy) on
    the plate domain (outside the hole).

    Args:
        R:          Hole radius.
        T:          Applied tension.
        E:          Young's modulus.
        nu:         Poisson's ratio.
        L:          Plate half-width.
        num_points: Number of validation points.
        seed:       Random seed.

    Returns:
        Dictionary with keys: 'coords', 'u', 'v', 'sigma_xx', 'sigma_yy', 'sigma_xy'
    """
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < num_points:
        pts = rng.uniform(-L, L, (num_points * 2, 2))
        valid = pts[np.sum(pts**2, axis=1) >= R**2]
        points.extend(valid.tolist())
    points = np.array(points[:num_points])

    x, y = points[:, 0], points[:, 1]
    sxx, syy, sxy = analytical_kirsch_stress(x, y, R, T)
    ux, uy = analytical_kirsch_displacement(x, y, R, T, E, nu)

    return {
        'coords': points,
        'u': ux,
        'v': uy,
        'sigma_xx': sxx,
        'sigma_yy': syy,
        'sigma_xy': sxy,
    }


if __name__ == "__main__":
    # Quick sanity check
    scf = stress_concentration_factor(R=0.2, T=1.0)
    print(f"Stress Concentration Factor (SCF) at hole edge: {scf:.4f}")
    assert abs(scf - 3.0) < 1e-10, f"SCF should be 3.0, got {scf}"
    print("✓ SCF = 3.0 verified.")

    # Evaluate along the critical line x=0
    y_line = np.linspace(0.2, 1.0, 50)
    x_line = np.zeros_like(y_line)
    sxx, syy, sxy = analytical_kirsch_stress(x_line, y_line, R=0.2, T=1.0)
    print(f"\nStress at hole edge (x=0, y=R=0.2):")
    print(f"  σ_xx = {sxx[0]:.4f}  (expected: 3.0 × T = 3.0)")
    print(f"  σ_yy = {syy[0]:.4f}  (expected: 0.0)")
    print(f"  σ_xy = {sxy[0]:.4f}  (expected: 0.0)")
