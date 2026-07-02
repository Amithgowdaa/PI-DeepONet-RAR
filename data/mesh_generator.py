"""
Mesh generation for 2D plates with circular holes/cutouts.

Uses the gmsh Python API for geometry definition, boolean operations,
and local mesh refinement. Converts to scikit-fem MeshTri format.

Two geometries:
  1. Quarter-symmetry plate with hole (Geometry 1)
  2. Full plate with circular cutout (Geometry 2)
"""

import numpy as np
import tempfile
import os

try:
    import gmsh
except ImportError:
    raise ImportError("gmsh is required. Install with: pip install gmsh")

try:
    import meshio
except ImportError:
    raise ImportError("meshio is required. Install with: pip install meshio")

try:
    from skfem import MeshTri
except ImportError:
    raise ImportError("scikit-fem is required. Install with: pip install scikit-fem")


def _initialize_gmsh(model_name: str = "model"):
    """Initialize gmsh, clearing any previous state."""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)  # suppress output
    gmsh.model.add(model_name)


def create_plate_with_hole_mesh(
    Lx: float = 2.0,
    Ly: float = 2.0,
    hole_center: tuple = (0.0, 0.0),
    hole_radius: float = 0.25,
    lc_bulk: float = 0.05,
    refinement_factor: int = 3,
) -> tuple:
    """
    Generate a quarter-symmetry mesh for a plate with circular hole.

    The quarter model covers [0, Lx] × [0, Ly] with a quarter-circle
    cutout at the origin. This exploits symmetry about both axes.

    Boundary tags returned:
        'left':   x = 0 (symmetry, u_x = 0)
        'right':  x = Lx (traction applied)
        'bottom': y = 0 (symmetry, u_y = 0)
        'top':    y = Ly (free)
        'hole':   circular arc (traction-free)

    Args:
        Lx, Ly: plate quarter dimensions [m]
        hole_center: center of full hole (should be origin for symmetry)
        hole_radius: hole radius [m]
        lc_bulk: mesh size far from hole [m]
        refinement_factor: mesh is this many times finer near hole

    Returns:
        (mesh, boundaries): skfem MeshTri and dict of boundary facet arrays
    """
    lc_hole = lc_bulk / refinement_factor

    _initialize_gmsh("plate_with_hole")

    # Use OpenCASCADE kernel for boolean operations
    occ = gmsh.model.occ

    # Rectangle: [0, Lx] × [0, Ly]
    rect = occ.addRectangle(0, 0, 0, Lx, Ly)

    # Quarter disk at origin
    disk = occ.addDisk(hole_center[0], hole_center[1], 0,
                       hole_radius, hole_radius)

    # Cut hole from rectangle
    result = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()

    # ── Mesh refinement near hole ──────────────────────────────────────────
    # Add a distance-based refinement field
    gmsh.model.mesh.field.add("Distance", 1)
    # Get all curves (edges) — find the hole arc
    curves = gmsh.model.getEntities(dim=1)
    hole_curves = []
    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        dist = np.sqrt((com[0] - hole_center[0])**2 +
                       (com[1] - hole_center[1])**2)
        if dist < hole_radius * 1.5:
            hole_curves.append(tag)

    gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    # Threshold field: fine near hole, coarse far away
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_hole)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_bulk)
    gmsh.model.mesh.field.setNumber(2, "DistMin", hole_radius * 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", hole_radius * 4.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Disable default meshing size constraints
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # ── Extract mesh and convert ───────────────────────────────────────────
    mesh, boundaries = _gmsh_to_skfem_with_boundaries(
        Lx, Ly, hole_center, hole_radius, geometry_type="quarter"
    )

    gmsh.finalize()
    return mesh, boundaries


def create_vessel_cutout_mesh(
    Lx: float = 3.0,
    Ly: float = 1.5,
    hole_center: tuple = (1.5, 0.75),
    hole_radius: float = 0.2,
    lc_bulk: float = 0.05,
    refinement_factor: int = 3,
) -> tuple:
    """
    Generate a full plate mesh with circular cutout (no symmetry).

    Boundary tags returned:
        'left':   x = 0 (pressure applied)
        'right':  x = Lx (pressure applied)
        'bottom': y = 0 (pressure applied)
        'top':    y = Ly (pressure applied)
        'hole':   circular arc (traction-free)

    Args:
        Lx, Ly: plate dimensions [m]
        hole_center: center of cutout [m]
        hole_radius: cutout radius [m]
        lc_bulk: mesh size far from cutout [m]
        refinement_factor: mesh is this many times finer near cutout

    Returns:
        (mesh, boundaries): skfem MeshTri and dict of boundary facet arrays
    """
    lc_hole = lc_bulk / refinement_factor

    _initialize_gmsh("vessel_cutout")

    occ = gmsh.model.occ

    # Full rectangle
    rect = occ.addRectangle(0, 0, 0, Lx, Ly)

    # Circular cutout
    disk = occ.addDisk(hole_center[0], hole_center[1], 0,
                       hole_radius, hole_radius)

    # Boolean cut
    result = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()

    # ── Mesh refinement near cutout ────────────────────────────────────────
    gmsh.model.mesh.field.add("Distance", 1)
    curves = gmsh.model.getEntities(dim=1)
    hole_curves = []
    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        dist = np.sqrt((com[0] - hole_center[0])**2 +
                       (com[1] - hole_center[1])**2)
        if dist < hole_radius * 1.5:
            hole_curves.append(tag)

    gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_hole)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_bulk)
    gmsh.model.mesh.field.setNumber(2, "DistMin", hole_radius * 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", hole_radius * 4.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    mesh, boundaries = _gmsh_to_skfem_with_boundaries(
        Lx, Ly, hole_center, hole_radius, geometry_type="full"
    )

    gmsh.finalize()
    return mesh, boundaries


def _gmsh_to_skfem_with_boundaries(
    Lx: float,
    Ly: float,
    hole_center: tuple,
    hole_radius: float,
    geometry_type: str = "quarter",
) -> tuple:
    """
    Extract mesh from active gmsh model, convert to skfem MeshTri,
    and identify boundary facets by geometric location.

    Args:
        Lx, Ly: domain dimensions
        hole_center: (cx, cy)
        hole_radius: r
        geometry_type: "quarter" or "full"

    Returns:
        (mesh, boundaries): MeshTri and dict mapping names to facet index arrays
    """
    # Save to temp file and read with meshio
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp_path = f.name
    gmsh.write(tmp_path)

    msh = meshio.read(tmp_path)
    os.unlink(tmp_path)

    # Extract triangle cells
    points_2d = msh.points[:, :2].T  # (2, N_nodes)

    tri_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "triangle":
            tri_cells = cell_block.data.T  # (3, N_elements)
            break

    if tri_cells is None:
        raise RuntimeError("No triangle cells found in gmsh mesh")

    mesh = MeshTri(points_2d, tri_cells)

    # ── Identify boundary facets ───────────────────────────────────────────
    # Facets are edges on the boundary
    boundary_facets = mesh.boundary_facets()
    facet_nodes = mesh.facets[:, boundary_facets]  # (2, n_boundary_facets)

    # Midpoint of each boundary facet
    mid_x = 0.5 * (mesh.p[0, facet_nodes[0]] + mesh.p[0, facet_nodes[1]])
    mid_y = 0.5 * (mesh.p[1, facet_nodes[0]] + mesh.p[1, facet_nodes[1]])

    # Geometric tolerances
    tol = Lx * 0.01  # 1% of domain size

    cx, cy = hole_center

    # Distance from facet midpoint to hole center
    dist_to_hole = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)

    boundaries = {}

    if geometry_type == "quarter":
        # Left: x ≈ 0
        boundaries['left'] = boundary_facets[mid_x < tol]
        # Right: x ≈ Lx
        boundaries['right'] = boundary_facets[np.abs(mid_x - Lx) < tol]
        # Bottom: y ≈ 0
        boundaries['bottom'] = boundary_facets[mid_y < tol]
        # Top: y ≈ Ly
        boundaries['top'] = boundary_facets[np.abs(mid_y - Ly) < tol]
        # Hole: distance to origin ≈ hole_radius
        boundaries['hole'] = boundary_facets[
            np.abs(dist_to_hole - hole_radius) < hole_radius * 0.3
        ]
    else:  # full
        boundaries['left'] = boundary_facets[mid_x < tol]
        boundaries['right'] = boundary_facets[np.abs(mid_x - Lx) < tol]
        boundaries['bottom'] = boundary_facets[mid_y < tol]
        boundaries['top'] = boundary_facets[np.abs(mid_y - Ly) < tol]
        boundaries['hole'] = boundary_facets[
            np.abs(dist_to_hole - hole_radius) < hole_radius * 0.3
        ]

    return mesh, boundaries


if __name__ == "__main__":
    print("Generating plate-with-hole mesh (quarter symmetry)...")
    mesh1, bd1 = create_plate_with_hole_mesh()
    print(f"  Nodes: {mesh1.p.shape[1]}, Elements: {mesh1.t.shape[1]}")
    for name, facets in bd1.items():
        print(f"  Boundary '{name}': {len(facets)} facets")

    print("\nGenerating vessel-cutout mesh (full plate)...")
    mesh2, bd2 = create_vessel_cutout_mesh()
    print(f"  Nodes: {mesh2.p.shape[1]}, Elements: {mesh2.t.shape[1]}")
    for name, facets in bd2.items():
        print(f"  Boundary '{name}': {len(facets)} facets")
