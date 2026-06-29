"""
FEniCS Data Generator for PI-DeepONet Ablation Study
====================================================
Generates Finite Element solution datasets for 2D Linear Elasticity problems:
1. Geometry 1: Square plate with circular hole under uniaxial tension.
2. Geometry 2: Rectangular pressure vessel plate with nozzle cutout under biaxial tension/pressure.

Exports CSV files formatted for DeepONet input/output matching grid points.
"""

import os
import sys
import numpy as np
import pandas as pd

# Standard Ubuntu Python site-packages/dist-packages check for Kaggle
ubuntu_py_path = "/usr/lib/python3/dist-packages"
if os.path.exists(ubuntu_py_path) and ubuntu_py_path not in sys.path:
    sys.path.append(ubuntu_py_path)

# Graceful FEniCS import check to prevent top-level script crashes
try:
    import dolfin
    import mshr
    HAS_FENICS = True
except ImportError:
    dolfin = None
    mshr = None
    HAS_FENICS = False

# ── 1. MATERIAL & CONSTITUTIVE DEFINITIONS ─────────────────────────────────
# Steel properties in MPa and meters
E_MODULUS = 200000.0  # 200 GPa = 200,000 MPa
POISSON_RATIO = 0.3

# Plane Stress Lame parameters
LMBDA = (E_MODULUS * POISSON_RATIO) / (1.0 - POISSON_RATIO**2)
MU = E_MODULUS / (2.0 * (1.0 + POISSON_RATIO))


def epsilon(u):
    """Symmetric strain tensor."""
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required for continuum strain tensor computations.")
    return dolfin.sym(dolfin.grad(u))


def sigma(u):
    """Stress tensor under 2D Plane Stress Hooke's Law."""
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required for constitutive stress calculations.")
    return LMBDA * dolfin.tr(epsilon(u)) * dolfin.Identity(2) + 2.0 * MU * epsilon(u)


def von_mises(s_xx, s_yy, s_xy):
    """Calculate 2D plane stress von Mises stress field."""
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)


# ── 2. MESH GENERATION & REFINEMENT ─────────────────────────────────────────
def generate_plate_with_hole_mesh(width=2.0, height=2.0, radius=0.25, base_res=35):
    """
    Generate mesh for Geometry 1 (Plate with hole) with 3x local refinement near hole.
    Domain centered at (0,0), spans [-1, 1] x [-1, 1].
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin/mshr) is required for mesh generation.")
        
    domain = mshr.Rectangle(
        dolfin.Point(-width / 2.0, -height / 2.0),
        dolfin.Point(width / 2.0, height / 2.0)
    ) - mshr.Circle(dolfin.Point(0.0, 0.0), radius)
    
    mesh = mshr.generate_mesh(domain, base_res)
    
    # 3x mesh refinement near the hole (within 2x radius)
    for _ in range(2):
        cell_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in dolfin.cells(mesh):
            p = cell.midpoint()
            r = np.sqrt(p.x()**2 + p.y()**2)
            if r < 2.0 * radius:
                cell_markers[cell] = True
        mesh = dolfin.refine(mesh, cell_markers)
        
    return mesh


def generate_pressure_vessel_mesh(width=3.0, height=1.5, radius=0.2, base_res=35):
    """
    Generate mesh for Geometry 2 (Pressure vessel nozzle cutout) with 3x local refinement.
    Domain centered at (0,0), spans [-1.5, 1.5] x [-0.75, 0.75].
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin/mshr) is required for mesh generation.")
        
    domain = mshr.Rectangle(
        dolfin.Point(-width / 2.0, -height / 2.0),
        dolfin.Point(width / 2.0, height / 2.0)
    ) - mshr.Circle(dolfin.Point(0.0, 0.0), radius)
    
    mesh = mshr.generate_mesh(domain, base_res)
    
    # Local refinement near cutout
    for _ in range(2):
        cell_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in dolfin.cells(mesh):
            p = cell.midpoint()
            r = np.sqrt(p.x()**2 + p.y()**2)
            if r < 2.5 * radius:
                cell_markers[cell] = True
        mesh = dolfin.refine(mesh, cell_markers)
        
    return mesh


# ── 3. GEOMETRY 1 SOLVER & BENCHMARK ───────────────────────────────────────
def solve_geometry_1(sigma_0, mesh):
    """
    Solve Geometry 1: Uniaxial tension sigma_0 on right edge x=1.
    Left edge x=-1 has symmetric BC (u_x = 0). Point pinned for y-motion.
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required to run the FEM solver.")
        
    V = dolfin.VectorFunctionSpace(mesh, "P", 2)
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    
    # Subdomain marking for right boundary
    class RightBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], 1.0)
            
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    RightBoundary().mark(boundaries, 1)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Boundary conditions
    class LeftBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], -1.0)
            
    class LeftCenterPoint(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], -1.0) and dolfin.near(x[1], 0.0, eps=0.05)

    bc_left = dolfin.DirichletBC(V.sub(0), dolfin.Constant(0.0), LeftBoundary())
    bc_pin = dolfin.DirichletBC(V.sub(1), dolfin.Constant(0.0), LeftCenterPoint(), method="pointwise")
    bcs = [bc_left, bc_pin]
    
    a = dolfin.inner(sigma(u), epsilon(v)) * dolfin.dx
    t_bc = dolfin.Constant((sigma_0, 0.0))
    L = dolfin.dot(t_bc, v) * ds(1)
    
    u_sol = dolfin.Function(V)
    dolfin.solve(a == L, u_sol, bcs)
    return u_sol


def compute_kirsch_l2_error(u_sol, mesh, sigma_0, radius=0.25):
    """
    Compute relative L2 stress error against Kirsch analytical solution.
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required for error computation.")
        
    W = dolfin.TensorFunctionSpace(mesh, "DG", 1)
    sigma_proj = dolfin.project(sigma(u_sol), W)
    
    # Grid evaluation points
    xs = np.linspace(-1.0, 1.0, 50)
    ys = np.linspace(-1.0, 1.0, 50)
    
    fem_stresses = []
    kirsch_stresses = []
    
    for x in xs:
        for y in ys:
            r = np.sqrt(x**2 + y**2)
            if r > radius + 1e-3:
                try:
                    s_mat = sigma_proj(dolfin.Point(x, y))
                    s_arr = np.asarray(s_mat).reshape(-1)
                    s_xx_fem = s_arr[0]
                    
                    theta = np.arctan2(y, x)
                    s_rr = (sigma_0 / 2.0) * (1.0 - (radius / r)**2) + \
                           (sigma_0 / 2.0) * (1.0 - 4.0 * (radius / r)**2 + 3.0 * (radius / r)**4) * np.cos(2.0 * theta)
                    s_tt = (sigma_0 / 2.0) * (1.0 + (radius / r)**2) - \
                           (sigma_0 / 2.0) * (1.0 + 3.0 * (radius / r)**4) * np.cos(2.0 * theta)
                    tau_rt = -(sigma_0 / 2.0) * (1.0 + 2.0 * (radius / r)**2 - 3.0 * (radius / r)**4) * np.sin(2.0 * theta)
                    
                    s_xx_kirsch = s_rr * np.cos(theta)**2 + s_tt * np.sin(theta)**2 - 2.0 * tau_rt * np.sin(theta) * np.cos(theta)
                    
                    fem_stresses.append(s_xx_fem)
                    kirsch_stresses.append(s_xx_kirsch)
                except Exception:
                    pass
                    
    fem_arr = np.array(fem_stresses)
    kirsch_arr = np.array(kirsch_stresses)
    if len(kirsch_arr) == 0:
        return 0.0
    l2_error = np.linalg.norm(fem_arr - kirsch_arr) / np.linalg.norm(kirsch_arr)
    return l2_error


# ── 4. GEOMETRY 2 SOLVER ────────────────────────────────────────────────────
def solve_geometry_2(p_val, mesh):
    """
    Solve Geometry 2: Biaxial pressure load p_val on outer boundaries.
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required to run the FEM solver.")
        
    V = dolfin.VectorFunctionSpace(mesh, "P", 2)
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    
    class LeftEdge(dolfin.SubDomain):
        def inside(self, x, on_b): return on_b and dolfin.near(x[0], -1.5)
    class RightEdge(dolfin.SubDomain):
        def inside(self, x, on_b): return on_b and dolfin.near(x[0], 1.5)
    class BottomEdge(dolfin.SubDomain):
        def inside(self, x, on_b): return on_b and dolfin.near(x[1], -0.75)
    class TopEdge(dolfin.SubDomain):
        def inside(self, x, on_b): return on_b and dolfin.near(x[1], 0.75)
        
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LeftEdge().mark(boundaries, 1)
    RightEdge().mark(boundaries, 2)
    BottomEdge().mark(boundaries, 3)
    TopEdge().mark(boundaries, 4)
    
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Fix symmetry points to remove rigid body motions
    class XPin(dolfin.SubDomain):
        def inside(self, x, on_b): return dolfin.near(x[0], -1.5) and dolfin.near(x[1], 0.0, eps=0.05)
    class YPin(dolfin.SubDomain):
        def inside(self, x, on_b): return dolfin.near(x[0], 0.0, eps=0.05) and dolfin.near(x[1], -0.75)
        
    bc_x = dolfin.DirichletBC(V.sub(0), dolfin.Constant(0.0), XPin(), method="pointwise")
    bc_y = dolfin.DirichletBC(V.sub(1), dolfin.Constant(0.0), YPin(), method="pointwise")
    bcs = [bc_x, bc_y]
    
    a = dolfin.inner(sigma(u), epsilon(v)) * dolfin.dx
    L = dolfin.dot(dolfin.Constant((-p_val, 0.0)), v) * ds(1) + \
        dolfin.dot(dolfin.Constant((p_val, 0.0)), v) * ds(2) + \
        dolfin.dot(dolfin.Constant((0.0, -p_val)), v) * ds(3) + \
        dolfin.dot(dolfin.Constant((0.0, p_val)), v) * ds(4)
        
    u_sol = dolfin.Function(V)
    dolfin.solve(a == L, u_sol, bcs)
    return u_sol


# ── 5. GRID INTERPOLATION & CSV EXPORT ──────────────────────────────────────
def extract_grid_dataset(u_sol, mesh, bounds, grid_size=50, radius=0.25, load_val=10.0):
    """
    Extract [x, y, sigma_xx, sigma_yy, sigma_xy, u_x, u_y, load_value] on grid.
    Points inside circular cutout are assigned NaN.
    """
    if not HAS_FENICS:
        raise RuntimeError("FEniCS (dolfin) is required for dataset grid extraction.")
        
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    
    W = dolfin.TensorFunctionSpace(mesh, "DG", 1)
    sigma_proj = dolfin.project(sigma(u_sol), W)
    
    records = []
    for x in xs:
        for y in ys:
            r = np.sqrt(x**2 + y**2)
            if r <= radius:
                records.append([x, y, np.nan, np.nan, np.nan, np.nan, np.nan, load_val])
            else:
                try:
                    pt = dolfin.Point(x, y)
                    u_val = u_sol(pt)
                    s_mat = sigma_proj(pt)
                    s_arr = np.asarray(s_mat).reshape(-1)
                    s_xx = s_arr[0]
                    s_yy = s_arr[3] if len(s_arr) > 3 else s_arr[1]
                    s_xy = s_arr[1] if len(s_arr) > 3 else 0.0
                    records.append([x, y, s_xx, s_yy, s_xy, u_val[0], u_val[1], load_val])
                except Exception:
                    records.append([x, y, np.nan, np.nan, np.nan, np.nan, np.nan, load_val])
                    
    df = pd.DataFrame(records, columns=["x", "y", "sigma_xx", "sigma_yy", "sigma_xy", "u_x", "u_y", "load_value"])
    return df
