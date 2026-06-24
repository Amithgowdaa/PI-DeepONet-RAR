# Running and Testing Guide: PI-DeepONet-RAR

This manual provides detailed instructions on how to set up the environment, run training experiments for all four experimental arms, run the FEM and analytical baselines, and execute the unit tests.

---

## 1. Environment Setup

### Option A: Using Conda (Recommended)
You can create a dedicated Conda environment using the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate pi_deeponet_rar
```

### Option B: Using Pip
Alternatively, you can install the dependencies directly using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 2. Running Training Experiments

The training script `src/train.py` reads configuration parameters from YAML files located in the `configs/` directory.

We compare four distinct configurations (experimental arms):

| Configuration | Config Path | Description |
| :--- | :--- | :--- |
| **Baseline** | `configs/baseline.yaml` | Uniform random sampling of domain points and load functions. |
| **Collocation RAR** | `configs/rar_collocation.yaml` | Adaptive refinement of collocation points in high PDE residual regions. |
| **Load RAR** | `configs/rar_load.yaml` | Adaptive refinement of the branch network's load function space. |
| **Combined RAR** | `configs/rar_combined.yaml` | Simultaneous adaptive refinement of both collocation points and load functions. |

### Commands to Run

To run the experiments, execute the following commands from the root directory:

```bash
# 1. Run Baseline (Uniform Sampling)
python src/train.py --config configs/baseline.yaml

# 2. Run Collocation-only RAR
python src/train.py --config configs/rar_collocation.yaml

# 3. Run Load-only RAR
python src/train.py --config configs/rar_load.yaml

# 4. Run Combined RAR (Collocation + Load)
python src/train.py --config configs/rar_combined.yaml
```

*Note: The results (trained model checkpoints) will be saved under the `results/` directory as `<experiment_name>_model.pt`.*

---

## 3. Running Baselines & Verification

### 3.1 Analytical Kirsch Verification
You can run a quick check of the exact analytical solution of the Kirsch problem (stress distribution around a circular hole in a plate under tension):

```bash
python fem_baseline/analytical_kirsch.py
```
This prints the exact stresses ($\sigma_{xx}, \sigma_{yy}, \sigma_{xy}$) at critical locations (e.g., at the hole boundary where the stress concentration factor is theoretically $3.0$).

### 3.2 Finite Element Method (FEM) Baseline
To solve the linear elasticity equations using FEniCS (ground truth for comparison):

```bash
python fem_baseline/kirsch_fem.py
```
*Note: FEniCS requires a Linux/WSL environment. If FEniCS/dolfin and mshr are not available, the script will print a graceful fallback message.*

---

## 4. Running Unit Tests

We have implemented a comprehensive suite of unit tests under the `tests/` directory to verify:
- Neural network architecture shapes (`FeedForward` and `PIDeepONet` models)
- Physics PDE residual calculation logic and automatic differentiation gradients
- Analytical Kirsch stress calculations
- Domain and boundary coordinate sampling bounds and hole constraint enforcement
- Data generation (`GaussianRandomField` sampling)

### Option A: Running the Zero-Dependency Test Runner (Recommended)
We provide a custom test runner that executes the entire test suite on any Python environment **without requiring external libraries** like `pytest` or `unittest` decorators:

```bash
python run_tests.py
```

### Option B: Running with Pytest
If you have `pytest` installed in your environment, you can run:

```bash
pytest
```

---

## 5. Visualizing & Analyzing Results

Use the Jupyter notebooks in the `notebooks/` directory to evaluate the model's accuracy:

1. **`notebooks/01_validate_fem.ipynb`**: Compares the FEM simulation against the analytical solution.
2. **`notebooks/02_plot_losses.ipynb`**: Compares the loss convergence histories across all 4 experimental arms.
3. **`notebooks/03_error_analysis.ipynb`**: Evaluates relative $L_2$ errors and predicts the Peak Stress Concentration Factor (SCF).

To launch the Jupyter notebook interface:
```bash
jupyter notebook
```
and navigate to the `notebooks/` folder.
