"""
Experiment: FEM Data Generation for PI-DeepONet Ablation Study

Run from repo root:
    python experiments/fem_datagen/run.py

Or call run() from Kaggle notebook / kaggle_run_all.py.

Generates CSV training datasets for two geometries:
  Geometry 1: Plate with circular hole — uniaxial tension
  Geometry 2: Pressure vessel nozzle cutout — biaxial pressure

Each CSV contains [x, y, sigma_xx, sigma_yy, sigma_xy, u_x, u_y, load_value]
on a 50×50 grid for each load case.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import numpy as np
import pandas as pd
import time

from data.solvers import solve_plate_with_hole, solve_vessel_cutout
from data.analytical import compute_kirsch_l2_error


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/fem_datagen.yaml")


def save_grid_csv(grid_data: dict, load_value: float, filepath: str):
    """
    Save grid data to CSV with columns:
    x, y, sigma_xx, sigma_yy, sigma_xy, u_x, u_y, load_value

    Args:
        grid_data: dict from solver with grid arrays
        load_value: applied load [Pa]
        filepath: output CSV path
    """
    df = pd.DataFrame({
        'x': grid_data['x'],
        'y': grid_data['y'],
        'sigma_xx': grid_data['sigma_xx'],
        'sigma_yy': grid_data['sigma_yy'],
        'sigma_xy': grid_data['sigma_xy'],
        'u_x': grid_data['u_x'],
        'u_y': grid_data['u_y'],
        'load_value': load_value,
    })

    # Drop NaN rows (points inside hole)
    df = df.dropna()
    df.to_csv(filepath, index=False)
    return df


def run(config_path: str = CONFIG_PATH, output_dir: str = None):
    """
    Main entry point: runs both geometries across all load cases.

    Args:
        config_path: path to fem_datagen.yaml
        output_dir: override output directory (default from config)
    """
    # ── Load config ────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mat = cfg['material']
    E = mat['E']
    nu = mat['nu']
    mesh_cfg = cfg['mesh']

    if output_dir is None:
        # Use Kaggle path if it exists, else local fallback
        kaggle_dir = cfg['output']['directory']
        local_dir = cfg['output']['local_directory']
        output_dir = kaggle_dir if os.path.exists(os.path.dirname(kaggle_dir) or '/kaggle') else local_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Material: E = {E/1e9:.0f} GPa, ν = {nu}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # GEOMETRY 1: Plate with Circular Hole
    # ═══════════════════════════════════════════════════════════════════════
    g1 = cfg['geometry1']
    loads_Pa = [s * 1e6 for s in g1['loads_MPa']]

    print(f"\n{'='*70}")
    print(f"GEOMETRY 1: Plate with Circular Hole")
    print(f"  Domain: {g1['Lx']}m × {g1['Ly']}m (quarter symmetry)")
    print(f"  Hole radius: {g1['hole_radius']}m")
    print(f"  Load cases: {g1['loads_MPa']} MPa")
    print(f"{'='*70}\n")

    for i, (sigma_mpa, sigma_pa) in enumerate(zip(g1['loads_MPa'], loads_Pa)):
        print(f"  [{i+1}/{len(loads_Pa)}] σ = {sigma_mpa} MPa ... ", end="", flush=True)

        result = solve_plate_with_hole(
            sigma_applied=sigma_pa,
            E=E, nu=nu,
            Lx=g1['Lx'], Ly=g1['Ly'],
            hole_radius=g1['hole_radius'],
            lc_bulk=mesh_cfg['lc_bulk'],
            refinement_factor=mesh_cfg['refinement_factor'],
            grid_nx=g1['grid_nx'], grid_ny=g1['grid_ny'],
        )

        # Save CSV
        filename = f"plate_hole_{sigma_mpa}MPa.csv"
        filepath = os.path.join(output_dir, filename)
        df = save_grid_csv(result, sigma_pa, filepath)

        # Kirsch validation
        errors = compute_kirsch_l2_error(
            x=result['x'], y=result['y'],
            fem_sxx=result['sigma_xx'],
            fem_syy=result['sigma_yy'],
            fem_sxy=result['sigma_xy'],
            a=g1['hole_radius'],
            sigma_inf=sigma_pa,
        )

        # Print summary
        print(f"done in {result['solve_time_s']:.1f}s")
        print(f"        Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")
        print(f"        CSV: {filename} ({len(df)} grid points)")
        print(f"        Max von Mises: {result['max_von_mises']/1e6:.2f} MPa "
              f"(SCF ≈ {result['max_von_mises']/sigma_pa:.2f})")
        print(f"        Kirsch L2 error: combined={errors['l2_combined']:.4f}, "
              f"σ_xx={errors['l2_sxx']:.4f}, σ_yy={errors['l2_syy']:.4f}, "
              f"σ_xy={errors['l2_sxy']:.4f}")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # GEOMETRY 2: Pressure Vessel Nozzle Cutout
    # ═══════════════════════════════════════════════════════════════════════
    g2 = cfg['geometry2']
    pressures_Pa = [p * 1e6 for p in g2['pressures_MPa']]

    print(f"\n{'='*70}")
    print(f"GEOMETRY 2: Pressure Vessel Nozzle Cutout")
    print(f"  Domain: {g2['Lx']}m × {g2['Ly']}m (full plate)")
    print(f"  Hole center: {g2['hole_center']}, radius: {g2['hole_radius']}m")
    print(f"  Pressure cases: {g2['pressures_MPa']} MPa")
    print(f"{'='*70}\n")

    for i, (p_mpa, p_pa) in enumerate(zip(g2['pressures_MPa'], pressures_Pa)):
        print(f"  [{i+1}/{len(pressures_Pa)}] p = {p_mpa} MPa ... ", end="", flush=True)

        result = solve_vessel_cutout(
            pressure=p_pa,
            E=E, nu=nu,
            Lx=g2['Lx'], Ly=g2['Ly'],
            hole_center=tuple(g2['hole_center']),
            hole_radius=g2['hole_radius'],
            lc_bulk=mesh_cfg['lc_bulk'],
            refinement_factor=mesh_cfg['refinement_factor'],
            grid_nx=g2['grid_nx'], grid_ny=g2['grid_ny'],
        )

        # Save CSV
        filename = f"vessel_{p_mpa}MPa.csv"
        filepath = os.path.join(output_dir, filename)
        df = save_grid_csv(result, p_pa, filepath)

        print(f"done in {result['solve_time_s']:.1f}s")
        print(f"        Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")
        print(f"        CSV: {filename} ({len(df)} grid points)")
        print(f"        Max von Mises: {result['max_von_mises']/1e6:.2f} MPa")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print(f"  Output directory: {output_dir}")
    n_files = len(g1['loads_MPa']) + len(g2['pressures_MPa'])
    print(f"  Total CSV files: {n_files}")
    print(f"  Grid per file: {g1['grid_nx']}×{g1['grid_ny']} = {g1['grid_nx']*g1['grid_ny']} points")
    print("=" * 70)


if __name__ == "__main__":
    run()
