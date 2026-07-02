"""
Kirsch Analytical Solution — Infinite plate with circular hole.

Classic elasticity benchmark for validating FEM stress fields.
Under uniaxial tension σ∞ in the x-direction, the exact stress field
around a circular hole of radius a is given by Kirsch (1898).

Reference: Timoshenko & Goodier, Theory of Elasticity, Ch. 4.
"""

import numpy as np


def kirsch_stress(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    sigma_inf: float,
) -> tuple:
    """
    Kirsch analytical stress at points (x, y) for hole radius a
    under uniaxial tension sigma_inf in x-direction.

    Args:
        x: x-coordinates (N,)
        y: y-coordinates (N,)
        a: hole radius [m]
        sigma_inf: far-field uniaxial stress [Pa]

    Returns:
        sigma_xx, sigma_yy, sigma_xy — each (N,) arrays [Pa]

    Notes:
        Points inside the hole (r < a) return NaN.
        The solution assumes an infinite plate; for finite plates,
        the FEM result will deviate near the outer boundary.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Avoid division by zero inside hole
    mask = r >= a
    r_safe = np.where(mask, r, np.inf)

    ratio2 = (a / r_safe) ** 2   # (a/r)^2
    ratio4 = ratio2 ** 2         # (a/r)^4

    cos2t = np.cos(2 * theta)
    cos4t = np.cos(4 * theta)
    sin2t = np.sin(2 * theta)
    sin4t = np.sin(4 * theta)

    # ── Kirsch stress components (polar → Cartesian) ────────────────────────
    # σ_rr
    sig_rr = (sigma_inf / 2.0) * (
        (1.0 - ratio2) + (1.0 - 4.0 * ratio2 + 3.0 * ratio4) * cos2t
    )
    # σ_θθ
    sig_tt = (sigma_inf / 2.0) * (
        (1.0 + ratio2) - (1.0 + 3.0 * ratio4) * cos2t
    )
    # τ_rθ
    sig_rt = -(sigma_inf / 2.0) * (
        (1.0 + 2.0 * ratio2 - 3.0 * ratio4) * sin2t
    )

    # ── Transform polar → Cartesian ────────────────────────────────────────
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2 = cos_t ** 2
    sin2 = sin_t ** 2
    cs = cos_t * sin_t

    sigma_xx = sig_rr * cos2 - 2.0 * sig_rt * cs + sig_tt * sin2
    sigma_yy = sig_rr * sin2 + 2.0 * sig_rt * cs + sig_tt * cos2
    sigma_xy = (sig_rr - sig_tt) * cs + sig_rt * (cos2 - sin2)

    # Mask points inside hole
    sigma_xx = np.where(mask, sigma_xx, np.nan)
    sigma_yy = np.where(mask, sigma_yy, np.nan)
    sigma_xy = np.where(mask, sigma_xy, np.nan)

    return sigma_xx, sigma_yy, sigma_xy


def compute_kirsch_l2_error(
    x: np.ndarray,
    y: np.ndarray,
    fem_sxx: np.ndarray,
    fem_syy: np.ndarray,
    fem_sxy: np.ndarray,
    a: float,
    sigma_inf: float,
) -> dict:
    """
    Compute relative L2 error between FEM and Kirsch analytical solution.

    Only compares points outside the hole AND sufficiently far from
    the outer boundary (where Kirsch infinite-plate assumption breaks down).

    Args:
        x, y: grid coordinates (N,)
        fem_sxx, fem_syy, fem_sxy: FEM stress fields (N,)
        a: hole radius [m]
        sigma_inf: applied far-field stress [Pa]

    Returns:
        dict with keys: 'l2_sxx', 'l2_syy', 'l2_sxy', 'l2_combined'
    """
    # Analytical
    ana_sxx, ana_syy, ana_sxy = kirsch_stress(x, y, a, sigma_inf)

    # Valid mask: outside hole AND not NaN
    valid = np.isfinite(ana_sxx) & np.isfinite(fem_sxx)

    # Exclude points too close to outer boundary (within 10% of domain edge)
    r = np.sqrt(x**2 + y**2)
    r_max = np.max(r[valid])
    inner_mask = r < 0.85 * r_max   # keep interior points only
    valid = valid & inner_mask

    if np.sum(valid) < 10:
        return {'l2_sxx': np.nan, 'l2_syy': np.nan,
                'l2_sxy': np.nan, 'l2_combined': np.nan}

    def rel_l2(fem_f, ana_f):
        diff = fem_f[valid] - ana_f[valid]
        norm_ana = np.linalg.norm(ana_f[valid])
        if norm_ana < 1e-12:
            return np.linalg.norm(diff)
        return np.linalg.norm(diff) / norm_ana

    err_sxx = rel_l2(fem_sxx, ana_sxx)
    err_syy = rel_l2(fem_syy, ana_syy)
    err_sxy = rel_l2(fem_sxy, ana_sxy)

    # Combined: stack all components
    fem_all = np.concatenate([fem_sxx[valid], fem_syy[valid], fem_sxy[valid]])
    ana_all = np.concatenate([ana_sxx[valid], ana_syy[valid], ana_sxy[valid]])
    norm_all = np.linalg.norm(ana_all)
    err_comb = np.linalg.norm(fem_all - ana_all) / norm_all if norm_all > 1e-12 else np.nan

    return {
        'l2_sxx': err_sxx,
        'l2_syy': err_syy,
        'l2_sxy': err_sxy,
        'l2_combined': err_comb,
    }


if __name__ == "__main__":
    # Quick test: stress at hole edge (r = a, theta = pi/2) should be 3*sigma
    a, sigma = 0.25, 100e6
    sxx, syy, sxy = kirsch_stress(
        np.array([0.0]), np.array([a]), a, sigma
    )
    print(f"At hole edge (0, a): σ_xx = {sxx[0]/1e6:.1f} MPa  (expect {3*sigma/1e6:.1f})")
    print(f"                     σ_yy = {syy[0]/1e6:.1f} MPa  (expect 0.0)")
