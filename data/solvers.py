"""
Problem-specific FEM solvers for 2D plane-stress elasticity.

Solver 1: Plate with circular hole — quarter-symmetry, uniaxial tension
Solver 2: Pressure vessel nozzle cutout — full plate, biaxial pressure

Both return stress/displacement fields on a uniform 50×50 grid,
ready for CSV export and DeepONet training.
"""

import numpy as np
import time

from skfem import (
    Basis, ElementTriP1, ElementVector, condense, solve,
)

from data.fem_setup import (
    assemble_stiffness,
    compute_stress_at_nodes,
    compute_von_mises,
    interpolate_to_grid,
)
from data.mesh_generator import (
    create_plate_with_hole_mesh,
    create_vessel_cutout_mesh,
)


# ============================================================================
# Solver 1: Plate with Circular Hole (Quarter Symmetry)
# ============================================================================

def solve_plate_with_hole(
    sigma_applied: float,
    E: float = 200e9,
    nu: float = 0.3,
    Lx: float = 2.0,
    Ly: float = 2.0,
    hole_radius: float = 0.25,
    lc_bulk: float = 0.05,
    refinement_factor: int = 3,
    grid_nx: int = 50,
    grid_ny: int = 50,
) -> dict:
    """
    Solve plate-with-hole under uniaxial tension (quarter-symmetry).

    BCs:
        - Left edge (x=0): u_x = 0 (symmetry)
        - Bottom edge (y=0): u_y = 0 (symmetry)
        - Right edge (x=Lx): σ_xx = sigma_applied (traction)
        - Top edge (y=Ly): free
        - Hole boundary: traction-free

    Args:
        sigma_applied: applied uniaxial stress [Pa] on right edge
        E: Young's modulus [Pa]
        nu: Poisson's ratio
        Lx, Ly: quarter-plate dimensions [m]
        hole_radius: radius [m]
        lc_bulk: bulk mesh size [m]
        refinement_factor: refinement near hole
        grid_nx, grid_ny: output grid resolution

    Returns:
        dict with keys: x, y, sigma_xx, sigma_yy, sigma_xy, u_x, u_y,
                        max_von_mises, n_nodes, n_elements, solve_time_s
    """
    t0 = time.time()

    # ── Mesh ───────────────────────────────────────────────────────────────
    mesh, boundaries = create_plate_with_hole_mesh(
        Lx=Lx, Ly=Ly,
        hole_center=(0.0, 0.0),
        hole_radius=hole_radius,
        lc_bulk=lc_bulk,
        refinement_factor=refinement_factor,
    )

    # ── Basis (vector P1 elements) ─────────────────────────────────────────
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)

    # ── Stiffness matrix ───────────────────────────────────────────────────
    K = assemble_stiffness(basis, E, nu)

    # ── Load vector: traction on right edge ─────────────────────────────────
    f = np.zeros(K.shape[0])

    # Apply σ_xx = sigma_applied on right edge via Neumann BC
    right_basis = basis.boundary(boundaries['right'])

    from skfem import LinearForm

    @LinearForm
    def traction_right(v, w):
        # Traction = [sigma_applied, 0] on right edge
        return sigma_applied * v[0]

    f += traction_right.assemble(right_basis)

    # ── Dirichlet BCs: symmetry ─────────────────────────────────────────────
    # u_x = 0 on left edge
    dofs_left_ux = basis.get_dofs(boundaries['left']).nodal['u^1']
    # u_y = 0 on bottom edge
    dofs_bottom_uy = basis.get_dofs(boundaries['bottom']).nodal['u^2']

    all_dirichlet = np.concatenate([dofs_left_ux, dofs_bottom_uy])

    # ── Solve ──────────────────────────────────────────────────────────────
    u = solve(*condense(K, f, D=all_dirichlet))

    # ── Post-process ───────────────────────────────────────────────────────
    sxx, syy, sxy, ux, uy = compute_stress_at_nodes(mesh, basis, u, E, nu)
    svm = compute_von_mises(sxx, syy, sxy)

    # ── Interpolate to grid ────────────────────────────────────────────────
    # Grid covers the quarter domain, excluding the hole
    grid_data = interpolate_to_grid(
        mesh.p,
        {'sigma_xx': sxx, 'sigma_yy': syy, 'sigma_xy': sxy,
         'u_x': ux, 'u_y': uy},
        grid_bounds=(0.0, Lx, 0.0, Ly),
        nx=grid_nx, ny=grid_ny,
        hole_center=(0.0, 0.0),
        hole_radius=hole_radius,
    )

    solve_time = time.time() - t0

    grid_data['max_von_mises'] = np.nanmax(svm)
    grid_data['n_nodes'] = mesh.p.shape[1]
    grid_data['n_elements'] = mesh.t.shape[1]
    grid_data['solve_time_s'] = solve_time

    return grid_data


# ============================================================================
# Solver 2: Pressure Vessel Nozzle Cutout (Full Plate)
# ============================================================================

def solve_vessel_cutout(
    pressure: float,
    E: float = 200e9,
    nu: float = 0.3,
    Lx: float = 3.0,
    Ly: float = 1.5,
    hole_center: tuple = (1.5, 0.75),
    hole_radius: float = 0.2,
    lc_bulk: float = 0.05,
    refinement_factor: int = 3,
    grid_nx: int = 50,
    grid_ny: int = 50,
) -> dict:
    """
    Solve pressure vessel cutout under biaxial pressure loading.

    BCs:
        - All outer edges: uniform pressure (biaxial tension) t = p * n
        - Hole boundary: traction-free
        - Pin one node to prevent rigid body motion

    Args:
        pressure: biaxial pressure [Pa] (positive = tension on outer edges)
        E: Young's modulus [Pa]
        nu: Poisson's ratio
        Lx, Ly: plate dimensions [m]
        hole_center: cutout center [m]
        hole_radius: cutout radius [m]
        lc_bulk: bulk mesh size [m]
        refinement_factor: refinement near cutout
        grid_nx, grid_ny: output grid resolution

    Returns:
        dict with same keys as solve_plate_with_hole
    """
    t0 = time.time()

    # ── Mesh ───────────────────────────────────────────────────────────────
    mesh, boundaries = create_vessel_cutout_mesh(
        Lx=Lx, Ly=Ly,
        hole_center=hole_center,
        hole_radius=hole_radius,
        lc_bulk=lc_bulk,
        refinement_factor=refinement_factor,
    )

    # ── Basis ──────────────────────────────────────────────────────────────
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)

    # ── Stiffness matrix ───────────────────────────────────────────────────
    K = assemble_stiffness(basis, E, nu)

    # ── Load vector: biaxial pressure on all outer edges ───────────────────
    f = np.zeros(K.shape[0])

    from skfem import LinearForm

    # Right edge: normal = [+1, 0], traction = [p, 0]
    @LinearForm
    def traction_right(v, w):
        return pressure * v[0]

    # Left edge: normal = [-1, 0], traction = [-p, 0]
    @LinearForm
    def traction_left(v, w):
        return -pressure * v[0]

    # Top edge: normal = [0, +1], traction = [0, p]
    @LinearForm
    def traction_top(v, w):
        return pressure * v[1]

    # Bottom edge: normal = [0, -1], traction = [0, -p]
    @LinearForm
    def traction_bottom(v, w):
        return -pressure * v[1]

    f += traction_right.assemble(basis.boundary(boundaries['right']))
    f += traction_left.assemble(basis.boundary(boundaries['left']))
    f += traction_top.assemble(basis.boundary(boundaries['top']))
    f += traction_bottom.assemble(basis.boundary(boundaries['bottom']))

    # ── Dirichlet BCs: pin corner to prevent rigid body motion ──────────────
    # Pin the bottom-left corner (0, 0) — fix both u_x and u_y
    corner_node = np.argmin(mesh.p[0]**2 + mesh.p[1]**2)

    # Get DOFs for this node
    dof_ux = basis.nodal_dofs[0, corner_node]
    dof_uy = basis.nodal_dofs[1, corner_node]

    # Also pin u_y of another node on the bottom edge to prevent rotation
    bottom_nodes = np.where(mesh.p[1] < Ly * 0.01)[0]
    if len(bottom_nodes) > 1:
        # Pick the rightmost bottom node
        far_node = bottom_nodes[np.argmax(mesh.p[0, bottom_nodes])]
        dof_uy_far = basis.nodal_dofs[1, far_node]
        all_dirichlet = np.array([dof_ux, dof_uy, dof_uy_far])
    else:
        all_dirichlet = np.array([dof_ux, dof_uy])

    # ── Solve ──────────────────────────────────────────────────────────────
    u = solve(*condense(K, f, D=all_dirichlet))

    # ── Post-process ───────────────────────────────────────────────────────
    sxx, syy, sxy, ux, uy = compute_stress_at_nodes(mesh, basis, u, E, nu)
    svm = compute_von_mises(sxx, syy, sxy)

    # ── Interpolate to grid ────────────────────────────────────────────────
    grid_data = interpolate_to_grid(
        mesh.p,
        {'sigma_xx': sxx, 'sigma_yy': syy, 'sigma_xy': sxy,
         'u_x': ux, 'u_y': uy},
        grid_bounds=(0.0, Lx, 0.0, Ly),
        nx=grid_nx, ny=grid_ny,
        hole_center=hole_center,
        hole_radius=hole_radius,
    )

    solve_time = time.time() - t0

    grid_data['max_von_mises'] = np.nanmax(svm)
    grid_data['n_nodes'] = mesh.p.shape[1]
    grid_data['n_elements'] = mesh.t.shape[1]
    grid_data['solve_time_s'] = solve_time

    return grid_data


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Plate with hole, σ = 100 MPa")
    print("=" * 60)
    result = solve_plate_with_hole(sigma_applied=100e6)
    print(f"  Grid points: {len(result['x'])}")
    print(f"  Max von Mises: {result['max_von_mises']/1e6:.1f} MPa")
    print(f"  Expected SCF ≈ 3.0 → {3*100:.1f} MPa")
    print(f"  Solve time: {result['solve_time_s']:.2f}s")

    print("\n" + "=" * 60)
    print("Test: Vessel cutout, p = 10 MPa")
    print("=" * 60)
    result2 = solve_vessel_cutout(pressure=10e6)
    print(f"  Grid points: {len(result2['x'])}")
    print(f"  Max von Mises: {result2['max_von_mises']/1e6:.1f} MPa")
    print(f"  Solve time: {result2['solve_time_s']:.2f}s")
