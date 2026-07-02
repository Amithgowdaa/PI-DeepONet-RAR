"""
Data generation package for PI-DeepONet ablation study.

FEM-based training data generators for 2D plane-stress elasticity:
  - Plate with circular hole (quarter symmetry)
  - Pressure vessel nozzle cutout (full plate)

Uses scikit-fem + gmsh (pure Python, Kaggle-compatible).
"""

from data.solvers import solve_plate_with_hole, solve_vessel_cutout
from data.analytical import kirsch_stress, compute_kirsch_l2_error

__all__ = [
    "solve_plate_with_hole",
    "solve_vessel_cutout",
    "kirsch_stress",
    "compute_kirsch_l2_error",
]
