# Physics-Informed DeepONet with Residual-based Adaptive Refinement (PI-DeepONet-RAR)

A research project investigating **where to add training data** in operator learning for solid mechanics — in the spatial domain, in the load function space, or both — using **residual-based active learning** to find the model's weak points.

## Problem Statement

> I'm building a neural network that learns to predict stress distributions for any load pattern on a plate with a hole. Instead of running a slow physics simulation every time, the net gives instant predictions. I'm testing where to add training data — in the 2D space, in the load parameter space, or both — using active learning to find the model's weak points.

## Architecture

```
┌──────────────────┐     ┌──────────────────┐
│   Branch Net     │     │    Trunk Net      │
│                  │     │                   │
│  T(y) at sensors │     │  Coords (x, y)    │
│  [100 sensors]   │     │  [2D input]       │
│       ↓          │     │       ↓           │
│  MLP: 100→128    │     │  MLP: 2→128       │
│       →128→100   │     │      →128→50      │
│       ↓          │     │       ↓           │
│  [P×2 features]  │     │  [P features]     │
└───────┬──────────┘     └───────┬───────────┘
        │         dot product    │
        └──────────┬─────────────┘
                   ↓
            u(x,y), v(x,y)
         [displacement fields]
                   ↓
         ┌─────────┴─────────┐
         │  Physics Loss     │
         │  ∇·σ = 0 (PDE)    │
         │  σ·n = 0 (hole)   │
         │  σ_xx = T (edge)  │
         └─────────┬─────────┘
                   ↓
         RAR: Add points/loads
         where residual is HIGH
```

## Experimental Arms

| Arm | Strategy | What adapts? |
|---|---|---|
| **Baseline** | Fixed uniform sampling | Nothing |
| **Collocation RAR** | Add spatial points at high-residual locations | Trunk input space (x, y) |
| **Load RAR** | Add load functions the model struggles with | Branch input space T(y) |
| **Combined RAR** | Both simultaneously | Both spaces |

## Project Structure

```
PI-DeepONet-RAR/
│
├── README.md                    # This file
├── LICENSE                      # Apache 2.0
├── requirements.txt             # pip install -r requirements.txt
├── kaggle_run_all.py            # 🚀 One-command Kaggle runner
├── run_tests.py                 # Test suite runner
│
├── configs/                     # YAML configs for the 4 arms
│   ├── baseline.yaml
│   ├── rar_collocation.yaml
│   ├── rar_load.yaml
│   └── rar_combined.yaml
│
├── src/                         # Core PyTorch code
│   ├── model.py                 # PIDeepONet architecture
│   ├── physics.py               # 2D elasticity PDE residuals
│   ├── data_generation.py       # GRF load function sampling
│   └── train.py                 # Training loop with RAR
│
├── fem_baseline/                # Ground truth
│   ├── analytical_kirsch.py     # Exact Kirsch solution (closed-form)
│   └── kirsch_fem.py            # FEniCS placeholder (not needed)
│
├── tests/                       # Unit tests
│   ├── test_model.py
│   ├── test_physics.py
│   ├── test_data_generation.py
│   └── test_analytical.py
│
├── notebooks/                   # Analysis & visualization
│   ├── 01_validate_fem.ipynb
│   ├── 02_plot_losses.ipynb
│   └── 03_error_analysis.ipynb
│
├── results/                     # Generated outputs (gitignored)
│   ├── *_losses.csv
│   ├── *_best.pt
│   └── plots/
│
└── report/                      # LaTeX report
    ├── main.tex
    └── references.bib
```

## Quick Start

### Option 1: Kaggle (Recommended)

1. Upload this repository as a Kaggle dataset
2. Create a new notebook and run:

```python
# Cell 1: Install dependencies
!pip install -q torch numpy matplotlib scipy pyyaml pandas

# Cell 2: Quick test run (~5 min)
%cd /kaggle/input/pi-deeponet-rar/
!python kaggle_run_all.py --quick

# Cell 3: Full run (~2-3 hours on GPU)
!python kaggle_run_all.py
```

### Option 2: Local

```bash
# Install
pip install -r requirements.txt

# Run tests
python run_tests.py

# Run individual experiments
python src/train.py --config configs/baseline.yaml
python src/train.py --config configs/rar_collocation.yaml
python src/train.py --config configs/rar_load.yaml
python src/train.py --config configs/rar_combined.yaml

# Or run all at once
python kaggle_run_all.py --quick  # Fast test
python kaggle_run_all.py          # Full experiment
```

### Option 3: Run specific arms

```bash
python kaggle_run_all.py --arms baseline rar_combined --quick
```

## Key Physics

**Kirsch Problem**: An infinite plate with a circular hole of radius R under uniaxial tension T.

- **Governing equations**: 2D Navier-Cauchy (linear elasticity, plane stress)
- **Boundary conditions**: Traction-free hole surface, applied tension on outer edge
- **Key result**: Stress Concentration Factor (SCF) = 3.0 at the hole boundary
- **Non-dimensionalized**: E = 1.0, ν = 0.3 (stresses normalized by T)

## References

1. Lu et al. "Learning nonlinear operators via DeepONet" — *Nature Machine Intelligence* (2021)
2. Wang et al. "Learning the solution operator of parametric PDEs with PI-DeepONets" — *Science Advances* (2021)
3. Raissi et al. "Physics-informed neural networks" — *J. Computational Physics* (2019)
4. Lu et al. "DeepXDE: A deep learning library for solving DEs" — *SIAM Review* (2021)
