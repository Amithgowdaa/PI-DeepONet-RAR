"""
Core FEM infrastructure for 2D plane-stress linear elasticity.

Uses scikit-fem (skfem) for variational form assembly and solving.
All functions are solver-agnostic — they work with any skfem mesh/basis.

Physics:
    Plane stress: σ_zz = 0, strain-stress via modified Hooke's law.
    Bilinear form: a(u,v) = ∫ σ(u) : ε(v) dΩ
    where σ = C : ε, C is the 2D plane-stress stiffness matrix.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve

try:
    from skfem import (
        Basis, ElementTriP1, ElementTriP2, ElementVector,
        BilinearForm, LinearForm, asm, condense, solve,
    )
    from skfem.helpers import ddot, sym_grad, dot, grad, transpose
except ImportError:
    raise ImportError(
        "scikit-fem is required. Install with: pip install scikit-fem"
    )


# ============================================================================
# Material: Plane Stress Constitutive Matrix
# ============================================================================

def plane_stress_C(E: float, nu: float) -> np.ndarray:
    """
    3×3 plane-stress stiffness matrix (Voigt notation).

    Maps strain vector [ε_xx, ε_yy, 2*ε_xy] to stress [σ_xx, σ_yy, σ_xy].

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio

    Returns:
        C: (3, 3) constitutive matrix
    """
    factor = E / (1.0 - nu**2)
    C = factor * np.array([
        [1.0,  nu,  0.0          ],
        [nu,   1.0, 0.0          ],
        [0.0,  0.0, (1.0 - nu)/2.0],
    ])
    return C


def lame_params_plane_stress(E: float, nu: float) -> tuple:
    """
    Effective Lamé parameters for plane stress.

    In plane stress, the effective lambda differs from 3D:
        λ_eff = E*ν / (1 - ν²)     (NOT E*ν / ((1+ν)(1-2ν)))
        μ = E / (2(1+ν))            (same as 3D)

    Args:
        E: Young's modulus [Pa]
        nu: Poisson's ratio

    Returns:
        (lam_eff, mu): effective Lamé parameters
    """
    lam_eff = E * nu / (1.0 - nu**2)
    mu = E / (2.0 * (1.0 + nu))
    return lam_eff, mu


# ============================================================================
# Bilinear Form Assembly
# ============================================================================

def assemble_stiffness(basis, E: float, nu: float):
    """
    Assemble the global stiffness matrix for 2D plane-stress elasticity.

    Uses the symmetric gradient (strain) and the plane-stress constitutive
    law via Lamé parameters.

    Args:
        basis: skfem Basis (vector-valued, 2D)
        E: Young's modulus [Pa]
        nu: Poisson's ratio

    Returns:
        K: sparse stiffness matrix (CSR)
    """
    lam, mu = lame_params_plane_stress(E, nu)

    @BilinearForm
    def stiffness(u, v, w):
        # σ(u) : ε(v) with plane-stress constitutive law
        # σ = 2μ ε(u) + λ tr(ε(u)) I
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        # trace of 2D strain tensor
        tr_eps_u = eps_u[0, 0] + eps_u[1, 1]
        # Full contraction: σ:ε = 2μ (ε:ε) + λ (tr ε_u)(tr ε_v)
        tr_eps_v = eps_v[0, 0] + eps_v[1, 1]
        return 2.0 * mu * ddot(eps_u, eps_v) + lam * tr_eps_u * tr_eps_v

    return stiffness.assemble(basis)


def assemble_traction(basis, facets, traction_vec: np.ndarray):
    """
    Assemble a traction (Neumann) boundary load vector.

    Args:
        basis: skfem Basis (vector-valued)
        facets: boundary facet indices where traction is applied
        traction_vec: [tx, ty] traction vector [Pa]

    Returns:
        f: load vector (sparse)
    """
    fb = basis.boundary(facets)

    @LinearForm
    def traction_form(v, w):
        # Apply constant traction on boundary
        return traction_vec[0] * v[0] + traction_vec[1] * v[1]

    return traction_form.assemble(fb)


def assemble_pressure_load(basis, facets, pressure: float, normals):
    """
    Assemble a pressure (normal traction) boundary load vector.

    Pressure acts inward: t = -p * n_outward = p * n_inward.
    For biaxial loading on outer edges, we apply t = p * n_outward
    (tension convention).

    Args:
        basis: skfem Basis (vector-valued)
        facets: boundary facet indices
        pressure: pressure magnitude [Pa] (positive = tension)
        normals: outward normal direction [nx, ny]

    Returns:
        f: load vector
    """
    fb = basis.boundary(facets)
    tx = pressure * normals[0]
    ty = pressure * normals[1]

    @LinearForm
    def pressure_form(v, w):
        return tx * v[0] + ty * v[1]

    return pressure_form.assemble(fb)


# ============================================================================
# Post-Processing: Stress Recovery
# ============================================================================

def compute_stress_at_nodes(
    mesh,
    basis,
    displacement: np.ndarray,
    E: float,
    nu: float,
) -> tuple:
    """
    Recover stress field (σ_xx, σ_yy, σ_xy) at mesh nodes from displacement.

    Uses strain = sym_grad(u) computed at element integration points,
    then projects back to nodes via L2 projection (averaging).

    Args:
        mesh: skfem Mesh
        basis: skfem Basis (vector-valued)
        displacement: (2*N_nodes,) displacement DOF vector
        E: Young's modulus [Pa]
        nu: Poisson's ratio

    Returns:
        sigma_xx, sigma_yy, sigma_xy: each (N_nodes,) arrays [Pa]
        ux, uy: displacement components (N_nodes,) each [m]
    """
    C = plane_stress_C(E, nu)
    n_nodes = mesh.p.shape[1]

    # Extract displacement components
    ux = displacement[basis.nodal_dofs[0]]
    uy = displacement[basis.nodal_dofs[1]]

    # Compute strain at integration points via basis interpolation
    # Use a scalar P1 basis for projecting stress back to nodes
    scalar_elem = ElementTriP1()
    scalar_basis = Basis(mesh, scalar_elem)

    # For each element, compute strain from displacement gradients
    # We'll use the mapping and basis function gradients directly
    sigma_xx_nodes = np.zeros(n_nodes)
    sigma_yy_nodes = np.zeros(n_nodes)
    sigma_xy_nodes = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)

    # Element connectivity
    elements = mesh.t  # (3, n_elements) for triangles

    for el_idx in range(mesh.t.shape[1]):
        # Node indices for this element
        nodes = elements[:, el_idx]
        # Node coordinates
        coords = mesh.p[:, nodes]  # (2, 3)

        # Element displacement
        ux_el = ux[nodes]   # (3,)
        uy_el = uy[nodes]   # (3,)

        # Compute B-matrix (strain-displacement) for linear triangle
        # For a 3-node triangle with nodes (x1,y1), (x2,y2), (x3,y3):
        x1, x2, x3 = coords[0]
        y1, y2, y3 = coords[1]

        # Area of triangle
        A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(A2) < 1e-30:
            continue
        inv_A2 = 1.0 / A2

        # Shape function gradients (constant over element for P1)
        dN1_dx = (y2 - y3) * inv_A2
        dN2_dx = (y3 - y1) * inv_A2
        dN3_dx = (y1 - y2) * inv_A2
        dN1_dy = (x3 - x2) * inv_A2
        dN2_dy = (x1 - x3) * inv_A2
        dN3_dy = (x2 - x1) * inv_A2

        # Strain (constant per element for P1)
        eps_xx = dN1_dx * ux_el[0] + dN2_dx * ux_el[1] + dN3_dx * ux_el[2]
        eps_yy = dN1_dy * uy_el[0] + dN2_dy * uy_el[1] + dN3_dy * uy_el[2]
        eps_xy = 0.5 * (
            (dN1_dy * ux_el[0] + dN2_dy * ux_el[1] + dN3_dy * ux_el[2]) +
            (dN1_dx * uy_el[0] + dN2_dx * uy_el[1] + dN3_dx * uy_el[2])
        )

        # Stress = C * [eps_xx, eps_yy, 2*eps_xy]
        strain_voigt = np.array([eps_xx, eps_yy, 2.0 * eps_xy])
        stress_voigt = C @ strain_voigt  # [σ_xx, σ_yy, σ_xy]

        # Accumulate to nodes (averaging)
        for local_idx in range(3):
            node_id = nodes[local_idx]
            sigma_xx_nodes[node_id] += stress_voigt[0]
            sigma_yy_nodes[node_id] += stress_voigt[1]
            sigma_xy_nodes[node_id] += stress_voigt[2]
            node_count[node_id] += 1.0

    # Average
    mask = node_count > 0
    sigma_xx_nodes[mask] /= node_count[mask]
    sigma_yy_nodes[mask] /= node_count[mask]
    sigma_xy_nodes[mask] /= node_count[mask]

    return sigma_xx_nodes, sigma_yy_nodes, sigma_xy_nodes, ux, uy


def compute_von_mises(sigma_xx, sigma_yy, sigma_xy):
    """
    Von Mises (equivalent) stress for plane stress.

    σ_vm = sqrt(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

    Args:
        sigma_xx, sigma_yy, sigma_xy: stress components (any shape)

    Returns:
        sigma_vm: von Mises stress (same shape)
    """
    return np.sqrt(
        sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3.0 * sigma_xy**2
    )


# ============================================================================
# Grid Interpolation
# ============================================================================

def interpolate_to_grid(
    node_coords: np.ndarray,
    node_values: dict,
    grid_bounds: tuple,
    nx: int = 50,
    ny: int = 50,
    hole_center: tuple = None,
    hole_radius: float = None,
) -> dict:
    """
    Interpolate FEM nodal fields onto a regular rectangular grid.

    Points inside the circular hole are masked as NaN.

    Args:
        node_coords: (2, N_nodes) mesh node coordinates
        node_values: dict of field_name -> (N_nodes,) arrays
        grid_bounds: (x_min, x_max, y_min, y_max)
        nx, ny: grid resolution
        hole_center: (cx, cy) or None
        hole_radius: radius or None

    Returns:
        dict with 'x', 'y' grids and interpolated field arrays (nx*ny,)
    """
    x_min, x_max, y_min, y_max = grid_bounds
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    # Source points
    src_pts = node_coords.T  # (N_nodes, 2)

    result = {
        'x': grid_pts[:, 0],
        'y': grid_pts[:, 1],
    }

    for name, vals in node_values.items():
        interp = griddata(src_pts, vals, grid_pts, method='linear')
        # Fill NaN at grid edges with nearest-neighbor
        nan_mask = np.isnan(interp)
        if np.any(nan_mask):
            nearest = griddata(src_pts, vals, grid_pts, method='nearest')
            interp[nan_mask] = nearest[nan_mask]
        result[name] = interp

    # Mask points inside hole
    if hole_center is not None and hole_radius is not None:
        cx, cy = hole_center
        dist = np.sqrt((result['x'] - cx)**2 + (result['y'] - cy)**2)
        inside = dist < hole_radius * 0.98  # small buffer
        for name in node_values:
            result[name][inside] = np.nan

    return result


if __name__ == "__main__":
    # Quick test: constitutive matrix
    C = plane_stress_C(200e9, 0.3)
    print("Plane stress C matrix (GPa):")
    print(C / 1e9)
    lam_val, mu_val = lame_params_plane_stress(200e9, 0.3)
    print(f"\nLame params: lam_eff={lam_val/1e9:.2f} GPa, mu={mu_val/1e9:.2f} GPa")
