# Physics-Informed DeepONet with Residual-based Adaptive Refinement (PI-DeepONet-RAR) for Solid Mechanics

This repository implements a Physics-Informed Deep Operator Network (PI-DeepONet) combined with Residual-based Adaptive Refinement (RAR) to solve 2D linear elasticity problems (Navier's equations). The physical benchmark is the classic **Kirsch Problem**: a plate with a circular hole under uniaxial tension.

## Project Structure

```
pi-deeponet-rar-solid-mechanics/
│
├── README.md                 # Project overview, how to run, results summary
├── LICENSE                   # Apache 2.0 License
├── environment.yml           # Conda environment file
├── requirements.txt          # Pip fallback
│
├── configs/                  # HYDRA config files for the 4 arms
│   ├── baseline.yaml         # Uniform sampling
│   ├── rar_collocation.yaml  # Collocation-only RAR
│   ├── rar_load.yaml         # Load-function-only RAR
│   └── rar_combined.yaml     # Combined RAR
│
├── src/                      # Core DeepXDE / PyTorch code
│   ├── __init__.py
│   ├── model.py              # PI-DeepONet architecture definition
│   ├── physics.py            # 2D Linear elasticity PDE residuals (Navier)
│   ├── data_generation.py    # Scripts to generate training data for Branch net
│   └── train.py              # Main training loop (reads from configs/)
│
├── fem_baseline/             # FEniCS scripts for ground truth
│   ├── kirsch_fem.py         # FEniCS solver for the plate with a hole
│   └── analytical_kirsch.py  # Python implementation of the exact Kirsch equations
│
├── notebooks/                # Jupyter notebooks for analysis and plotting
│   ├── 01_validate_fem.ipynb # Plots FEM vs Analytical Kirsch
│   ├── 02_plot_losses.ipynb  # Plots training loss curves for all 4 arms
│   └── 03_error_analysis.ipynb # Calculates L2 relative errors, stress concentration factors
│
└── report/                   # LaTeX source for technical report
    ├── main.tex
    └── references.bib
```

## Methodology & Experimental Arms

To assess the performance of adaptive training strategies in operator learning for mechanics, we run four distinct configurations (arms):

1. **Baseline**: Training with a fixed, uniform distribution of collocation points and loading functions.
2. **RAR Collocation**: Iteratively adds collocation points in regions of high PDE residuals (stress concentrations).
3. **RAR Load**: Iteratively refines the load function space, adding function samples where predictions exhibit high residuals.
4. **RAR Combined**: Performs adaptive refinement on both coordinates (collocation) and load functions (branch net inputs) simultaneously.

## Getting Started

### Installation
Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate pi_deeponet_rar
```
Or use pip:
```bash
pip install -r requirements.txt
```

### Running Experiments
Use the main training entrypoint with Hydra configurations:
```bash
# Run baseline
python src/train.py --config-name baseline

# Run collocation-only RAR
python src/train.py --config-name rar_collocation

# Run load-only RAR
python src/train.py --config-name rar_load

# Run combined RAR
python src/train.py --config-name rar_combined
```
