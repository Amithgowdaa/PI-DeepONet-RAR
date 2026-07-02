"""
PI-DeepONet FEM Data Generation -- Kaggle Runner
=================================================

USAGE ON KAGGLE:
  1. Upload PI-DeepONet-RAR repo as a Kaggle Dataset (from GitHub)
  2. In a new notebook, add that dataset as input
  3. Copy-paste this entire file into a single cell and run

The script auto-detects:
  - Whether it is running on Kaggle or locally
  - The dataset input path (scans /kaggle/input/ for our package)
  - Installs missing pip dependencies on Kaggle automatically

Structure (designed as notebook cells):
  Cell 1: Install dependencies + path setup
  Cell 2: Imports
  Cell 3: Material + mesh setup functions
  Cell 4: Geometry 1 solver loop (plate with hole)
  Cell 5: Geometry 2 solver loop (vessel cutout)
  Cell 6: CSV export + validation prints

All output goes to /kaggle/working/ (Kaggle) or ./results/fem_datagen/ (local).
"""

import subprocess
import sys
import os
import glob


# ============================================================================
# CELL 1: INSTALL DEPENDENCIES + PATH SETUP
# ============================================================================
# scikit-fem: pure-Python FEM library (no C++ compilation needed)
# gmsh: industry-standard mesh generator with Python API
# meshio: mesh format converter (gmsh -> scikit-fem)
# Note: FEniCS CANNOT be installed on Kaggle due to C++ dependency chain.

ON_KAGGLE = os.path.exists("/kaggle/working")

def install_dependencies():
    """Install FEM dependencies. Run this once at notebook start."""
    packages = ["scikit-fem", "gmsh", "meshio", "pyyaml", "pandas"]
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", pkg
        ])
    print("All dependencies installed.")


def find_repo_root():
    """
    Auto-detect the repo root directory.

    On Kaggle: scans /kaggle/input/*/  for a directory containing data/solvers.py
    Locally:   uses __file__ directory (when run as a .py script)
    """
    if ON_KAGGLE:
        # Kaggle datasets are mounted under /kaggle/input/<dataset-name>/
        # The dataset name is auto-generated from the GitHub repo name.
        # Scan all input datasets for our package marker file.
        search_pattern = "/kaggle/input/*"
        for candidate in sorted(glob.glob(search_pattern)):
            marker = os.path.join(candidate, "data", "solvers.py")
            if os.path.isfile(marker):
                print(f"[setup] Found repo at: {candidate}")
                return candidate
        # Fallback: maybe a subdirectory (some datasets nest one level deeper)
        search_pattern_nested = "/kaggle/input/*/*"
        for candidate in sorted(glob.glob(search_pattern_nested)):
            marker = os.path.join(candidate, "data", "solvers.py")
            if os.path.isfile(marker):
                print(f"[setup] Found repo at: {candidate}")
                return candidate
        raise FileNotFoundError(
            "Could not find PI-DeepONet-RAR dataset in /kaggle/input/. "
            "Make sure you added the dataset as input to this notebook. "
            "Expected to find data/solvers.py inside the dataset."
        )
    else:
        # Local run: use __file__ if available, else current directory
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ not defined (e.g., interactive Python / Jupyter)
            return os.getcwd()


# -- Auto-install on Kaggle --------------------------------------------------
if ON_KAGGLE:
    install_dependencies()

# -- Set up import path ------------------------------------------------------
REPO_ROOT = find_repo_root()
sys.path.insert(0, REPO_ROOT)
print(f"[setup] REPO_ROOT = {REPO_ROOT}")


# ============================================================================
# CELL 2: IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import time

from data.solvers import solve_plate_with_hole, solve_vessel_cutout
from data.analytical import compute_kirsch_l2_error


# ============================================================================
# CELL 3: MATERIAL + MESH CONFIGURATION
# ============================================================================

# Steel properties
E = 200.0e9      # Young's modulus [Pa] = 200 GPa
NU = 0.3         # Poisson's ratio

# Mesh parameters
LC_BULK = 0.05   # characteristic length far from hole [m]
REFINEMENT = 3   # 3x finer near hole boundary

# Output grid
GRID_NX = 50
GRID_NY = 50

# Output directory: always write to /kaggle/working/ on Kaggle
OUTPUT_DIR = "/kaggle/working/" if ON_KAGGLE else os.path.join(REPO_ROOT, "results", "fem_datagen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Environment: {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Material: E = {E/1e9:.0f} GPa, nu = {NU}")
print(f"Mesh: lc_bulk = {LC_BULK}m, refinement = {REFINEMENT}x")
print(f"Grid: {GRID_NX}x{GRID_NY} = {GRID_NX*GRID_NY} points per case")


def save_csv(grid_data: dict, load_value_pa: float, filepath: str) -> pd.DataFrame:
    """Save grid data to CSV, dropping NaN rows (inside hole)."""
    df = pd.DataFrame({
        'x': grid_data['x'],
        'y': grid_data['y'],
        'sigma_xx': grid_data['sigma_xx'],
        'sigma_yy': grid_data['sigma_yy'],
        'sigma_xy': grid_data['sigma_xy'],
        'u_x': grid_data['u_x'],
        'u_y': grid_data['u_y'],
        'load_value': load_value_pa,
    })
    df = df.dropna()
    df.to_csv(filepath, index=False)
    return df


# ============================================================================
# CELL 4: GEOMETRY 1 — PLATE WITH CIRCULAR HOLE
# ============================================================================
# Square plate 2m × 2m (quarter symmetry), hole radius 0.25m at origin.
# Uniaxial tension on right edge, symmetry BCs on left/bottom.
# Top and hole are traction-free.

def run_geometry1():
    """Solve Geometry 1 for all load cases."""
    LX, LY = 2.0, 2.0
    HOLE_RADIUS = 0.25
    LOADS_MPA = [10, 25, 50, 75, 100]

    print("\n" + "=" * 70)
    print("GEOMETRY 1: Plate with Circular Hole (Quarter Symmetry)")
    print(f"  Domain: {LX}m x {LY}m, hole radius: {HOLE_RADIUS}m")
    print(f"  Load cases: {LOADS_MPA} MPa")
    print("=" * 70)

    results_g1 = []

    for i, sigma_mpa in enumerate(LOADS_MPA):
        sigma_pa = sigma_mpa * 1e6
        print(f"\n  [{i+1}/{len(LOADS_MPA)}] sigma = {sigma_mpa} MPa")

        result = solve_plate_with_hole(
            sigma_applied=sigma_pa,
            E=E, nu=NU,
            Lx=LX, Ly=LY,
            hole_radius=HOLE_RADIUS,
            lc_bulk=LC_BULK,
            refinement_factor=REFINEMENT,
            grid_nx=GRID_NX, grid_ny=GRID_NY,
        )

        # Save CSV
        filename = f"plate_hole_{sigma_mpa}MPa.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df = save_csv(result, sigma_pa, filepath)

        # Kirsch analytical validation
        errors = compute_kirsch_l2_error(
            x=result['x'], y=result['y'],
            fem_sxx=result['sigma_xx'],
            fem_syy=result['sigma_yy'],
            fem_sxy=result['sigma_xy'],
            a=HOLE_RADIUS,
            sigma_inf=sigma_pa,
        )

        # Print results
        scf = result['max_von_mises'] / sigma_pa
        print(f"    [OK] Saved: {filename} ({len(df)} points)")
        print(f"    [OK] Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")
        print(f"    [OK] Max von Mises: {result['max_von_mises']/1e6:.2f} MPa (SCF ~ {scf:.2f})")
        print(f"    [OK] Kirsch L2 error: {errors['l2_combined']:.4f} "
              f"(sxx: {errors['l2_sxx']:.4f}, syy: {errors['l2_syy']:.4f})")
        print(f"    [OK] Time: {result['solve_time_s']:.1f}s")

        results_g1.append({
            'load_MPa': sigma_mpa,
            'max_vm_MPa': result['max_von_mises'] / 1e6,
            'scf': scf,
            'kirsch_l2': errors['l2_combined'],
            'n_points': len(df),
        })

    return results_g1


# ============================================================================
# CELL 5: GEOMETRY 2 — PRESSURE VESSEL NOZZLE CUTOUT
# ============================================================================
# Plate 3m x 1.5m (full plate), circular cutout at center, radius 0.2m.
# Biaxial pressure on all outer edges, hole traction-free.

def run_geometry2():
    """Solve Geometry 2 for all pressure cases."""
    LX, LY = 3.0, 1.5
    HOLE_CENTER = (1.5, 0.75)
    HOLE_RADIUS = 0.2
    PRESSURES_MPA = [1, 2, 5, 10, 20]

    print("\n" + "=" * 70)
    print("GEOMETRY 2: Pressure Vessel Nozzle Cutout (Full Plate)")
    print(f"  Domain: {LX}m x {LY}m, hole center: {HOLE_CENTER}, radius: {HOLE_RADIUS}m")
    print(f"  Pressure cases: {PRESSURES_MPA} MPa")
    print("=" * 70)

    results_g2 = []

    for i, p_mpa in enumerate(PRESSURES_MPA):
        p_pa = p_mpa * 1e6
        print(f"\n  [{i+1}/{len(PRESSURES_MPA)}] p = {p_mpa} MPa")

        result = solve_vessel_cutout(
            pressure=p_pa,
            E=E, nu=NU,
            Lx=LX, Ly=LY,
            hole_center=HOLE_CENTER,
            hole_radius=HOLE_RADIUS,
            lc_bulk=LC_BULK,
            refinement_factor=REFINEMENT,
            grid_nx=GRID_NX, grid_ny=GRID_NY,
        )

        # Save CSV
        filename = f"vessel_{p_mpa}MPa.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df = save_csv(result, p_pa, filepath)

        print(f"    [OK] Saved: {filename} ({len(df)} points)")
        print(f"    [OK] Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")
        print(f"    [OK] Max von Mises: {result['max_von_mises']/1e6:.2f} MPa")
        print(f"    [OK] Time: {result['solve_time_s']:.1f}s")

        results_g2.append({
            'pressure_MPa': p_mpa,
            'max_vm_MPa': result['max_von_mises'] / 1e6,
            'n_points': len(df),
        })

    return results_g2


# ============================================================================
# CELL 6: RUN ALL + SUMMARY
# ============================================================================

def run_all():
    """Run both geometries and print final summary."""
    t_total = time.time()

    print("=" * 70)
    print("PI-DeepONet FEM DATA GENERATION")
    print("=" * 70)

    # Geometry 1
    results_g1 = run_geometry1()

    # Geometry 2
    results_g2 = run_geometry2()

    # ── Final Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_total

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Geometry 1 - Plate with Hole (Uniaxial Tension):")
    print(f"  {'Load (MPa)':>12} {'Max VM (MPa)':>14} {'SCF':>8} {'Kirsch L2':>12} {'Points':>8}")
    print(f"  {'-'*58}")
    for r in results_g1:
        print(f"  {r['load_MPa']:>12} {r['max_vm_MPa']:>14.2f} {r['scf']:>8.2f} "
              f"{r['kirsch_l2']:>12.4f} {r['n_points']:>8}")

    print(f"\n  Geometry 2 - Vessel Cutout (Biaxial Pressure):")
    print(f"  {'Pressure (MPa)':>15} {'Max VM (MPa)':>14} {'Points':>8}")
    print(f"  {'-'*40}")
    for r in results_g2:
        print(f"  {r['pressure_MPa']:>15} {r['max_vm_MPa']:>14.2f} {r['n_points']:>8}")

    # List output files
    csv_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
    print(f"\n  Output files ({len(csv_files)}):")
    for f in csv_files:
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    {f} ({size_kb:.0f} KB)")

    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
