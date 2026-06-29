"""
PI-DeepONet Training Loop with Residual-Based Adaptive Refinement (RAR)
========================================================================

Main training entrypoint. Reads configuration from YAML files and runs
one of four experimental arms:

    1. Baseline:       Uniform sampling (no adaptation)
    2. Collocation RAR: Adds collocation points where PDE residual is high
    3. Load RAR:       Adds load functions where prediction residual is high
    4. Combined RAR:   Adapts both collocation points and load functions

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/rar_collocation.yaml
    python src/train.py --config configs/rar_load.yaml
    python src/train.py --config configs/rar_combined.yaml

    # Quick mode for Kaggle testing (reduced scale)
    python src/train.py --config configs/baseline.yaml --quick
"""

import os
import sys
import argparse
import time
import csv
import yaml
import torch
import torch.optim as optim
import numpy as np

# ─── Fix import path for both local and Kaggle execution ───
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model import PIDeepONet
from physics import compute_pde_residuals, compute_stresses
from data_generation import GaussianRandomField


# ═══════════════════════════════════════════════════════════════
#  Domain & Boundary Sampling
# ═══════════════════════════════════════════════════════════════

def sample_domain_points(num_points, L=1.0, R=0.2, device='cpu'):
    """Samples points inside the plate domain (outside the circular hole)."""
    points = []
    while len(points) < num_points:
        pts = np.random.uniform(-L, L, (num_points * 2, 2))
        # Keep points outside the hole
        valid = pts[np.sum(pts**2, axis=1) >= R**2]
        points.extend(valid.tolist())
    coords = np.array(points[:num_points])
    return torch.tensor(coords, dtype=torch.float32, requires_grad=True, device=device)


def sample_boundary_points(num_points, L=1.0, R=0.2, device='cpu'):
    """
    Samples points along the boundaries:
        - Inner circle (traction-free): first half of returned points
        - Outer edges (applied loads):  second half of returned points
    """
    # 1. Inner circle boundary (traction free)
    num_inner = num_points // 2
    theta = np.random.uniform(0, 2 * np.pi, num_inner)
    inner_pts = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)

    # 2. Outer boundaries: x=±L, y=±L
    num_outer = num_points - num_inner
    outer_pts = []
    for _ in range(num_outer):
        side = np.random.choice(['left', 'right', 'bottom', 'top'])
        if side == 'left':
            outer_pts.append([-L, np.random.uniform(-L, L)])
        elif side == 'right':
            outer_pts.append([L, np.random.uniform(-L, L)])
        elif side == 'bottom':
            outer_pts.append([np.random.uniform(-L, L), -L])
        elif side == 'top':
            outer_pts.append([np.random.uniform(-L, L), L])

    points = np.vstack([inner_pts, np.array(outer_pts)])
    return torch.tensor(points, dtype=torch.float32, requires_grad=True, device=device)


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    """Auto-detect GPU availability."""
    device_str = config.get('training', {}).get('device', 'auto')
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


# ═══════════════════════════════════════════════════════════════
#  Loss Logger
# ═══════════════════════════════════════════════════════════════

class LossLogger:
    """Logs training losses to CSV for later plotting."""

    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'epoch', 'total_loss', 'pde_loss', 'bc_loss',
            'num_domain_pts', 'num_loads', 'elapsed_sec'
        ])
        self.start_time = time.time()

    def log(self, epoch, total_loss, pde_loss, bc_loss, num_dom, num_loads):
        elapsed = time.time() - self.start_time
        self.writer.writerow([
            epoch,
            f"{total_loss:.6e}",
            f"{pde_loss:.6e}",
            f"{bc_loss:.6e}",
            num_dom,
            num_loads,
            f"{elapsed:.1f}",
        ])
        self.file.flush()

    def close(self):
        self.file.close()


# ═══════════════════════════════════════════════════════════════
#  RAR Refinement
# ═══════════════════════════════════════════════════════════════

def rar_add_collocation_points(model, loads, coords_dom, config, L, R, device):
    """
    Evaluate PDE residual on candidate points and add those with the
    highest residual magnitude to the training set.

    IMPORTANT: We must enable gradients on the candidate coordinates
    because compute_pde_residuals() uses torch.autograd.grad() internally.
    We do NOT wrap this in torch.no_grad().
    """
    num_to_add = config['rar'].get('num_points_to_add', 50)
    num_eval_loads = min(5, loads.shape[0])  # Average over several loads

    # Generate candidate points
    coords_candidate = sample_domain_points(2000, L, R, device=device)

    total_residual = torch.zeros(coords_candidate.shape[0], device=device)

    for i in range(num_eval_loads):
        # Forward pass — need gradients for PDE residual computation
        outputs_candidate = model(loads[i:i + 1], coords_candidate)
        u_cand = outputs_candidate[0, :, 0:1]
        v_cand = outputs_candidate[0, :, 1:2]

        res_x, res_y, _ = compute_pde_residuals(coords_candidate, u_cand, v_cand)
        res_mag = (res_x**2 + res_y**2).squeeze()
        total_residual += res_mag.detach()

    avg_residual = total_residual / num_eval_loads

    # Find indices of top-residual points
    top_indices = torch.argsort(avg_residual, descending=True)[:num_to_add]
    new_points = coords_candidate[top_indices].detach().clone()

    # Concatenate with existing domain points
    coords_dom_new = torch.cat(
        [coords_dom.detach(), new_points], dim=0
    ).requires_grad_(True)

    return coords_dom_new


def rar_add_load_functions(model, loads, coords_dom, grf, config, device):
    """
    Generate candidate load functions, evaluate their average PDE residual,
    and add the most challenging ones to the training set.
    """
    num_to_add = config['rar'].get('num_loads_to_add', 10)
    num_candidates = 50

    # Sample candidate loads
    new_loads_np = grf.sample(num_candidates)
    new_loads = torch.tensor(new_loads_np, dtype=torch.float32, device=device)

    load_residuals = []

    for j in range(num_candidates):
        # Forward pass with gradients enabled for PDE residual
        outputs = model(new_loads[j:j + 1], coords_dom)
        u_l = outputs[0, :, 0:1]
        v_l = outputs[0, :, 1:2]

        res_x, res_y, _ = compute_pde_residuals(coords_dom, u_l, v_l)
        mean_res = torch.mean(res_x**2 + res_y**2).item()
        load_residuals.append(mean_res)

    # Select loads with highest average residual
    top_indices = np.argsort(load_residuals)[-num_to_add:]
    new_active_loads = new_loads[top_indices].detach()

    loads_new = torch.cat([loads, new_active_loads], dim=0)
    return loads_new


# ═══════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════

def _interpolate_load_to_y(load_vals, sensor_y, target_y):
    """
    Linearly interpolates the discretized load function T(y) to
    arbitrary y-coordinates on the boundary.

    Args:
        load_vals: (num_sensors,) tensor — load values at sensor locations.
        sensor_y:  (num_sensors,) tensor — sensor y-coordinates (sorted).
        target_y:  (M, 1) tensor — y-coordinates of boundary points.

    Returns:
        (M, 1) tensor of interpolated T(y) values.
    """
    target_flat = target_y.squeeze()  # (M,)
    # Clamp to sensor range to avoid extrapolation
    target_clamped = torch.clamp(target_flat, sensor_y[0], sensor_y[-1])

    # Find bin indices
    indices = torch.searchsorted(sensor_y, target_clamped).clamp(1, len(sensor_y) - 1)
    y_lo = sensor_y[indices - 1]
    y_hi = sensor_y[indices]
    v_lo = load_vals[indices - 1]
    v_hi = load_vals[indices]

    # Linear interpolation weight
    t = (target_clamped - y_lo) / (y_hi - y_lo + 1e-12)
    result = v_lo + t * (v_hi - v_lo)
    return result.unsqueeze(1)  # (M, 1)


def compute_losses(model, loads, coords_dom, coords_bnd, sensor_y, R, L):
    """
    Computes PDE residual loss and boundary condition loss.

    Args:
        model:      PIDeepONet model.
        loads:      (N_loads, num_sensors) tensor of load functions.
        coords_dom: (N_dom, 2) collocation coordinates (requires_grad=True).
        coords_bnd: (N_bnd, 2) boundary coordinates (requires_grad=True).
        sensor_y:   (num_sensors,) tensor of GRF sensor y-coordinates.
        R:          Hole radius.
        L:          Plate half-width.

    Returns:
        loss_pde: Mean PDE equilibrium residual squared.
        loss_bnd: Mean boundary condition residual squared.
    """
    num_loads = loads.shape[0]
    N_bnd = coords_bnd.shape[0]
    num_inner = N_bnd // 2  # first half = inner hole, second half = outer edges

    # Forward pass in Cartesian product mode
    outputs_dom = model(loads, coords_dom)   # (N_loads, N_dom, 2)
    outputs_bnd = model(loads, coords_bnd)   # (N_loads, N_bnd, 2)

    loss_pde = torch.tensor(0.0, device=loads.device)
    loss_bnd = torch.tensor(0.0, device=loads.device)

    for i in range(num_loads):
        # ─── PDE Loss ───
        u_dom = outputs_dom[i, :, 0:1]
        v_dom = outputs_dom[i, :, 1:2]
        res_x, res_y, _ = compute_pde_residuals(coords_dom, u_dom, v_dom)
        loss_pde = loss_pde + torch.mean(res_x**2 + res_y**2)

        # ─── Boundary Loss (uses lightweight stress computation) ───
        u_bnd = outputs_bnd[i, :, 0:1]
        v_bnd = outputs_bnd[i, :, 1:2]
        stresses_bnd = compute_stresses(coords_bnd, u_bnd, v_bnd)
        sig_xx = stresses_bnd['sigma_xx']
        sig_yy = stresses_bnd['sigma_yy']
        sig_xy = stresses_bnd['sigma_xy']

        # 1. Inner hole boundary (first half): traction-free → σ·n = 0
        inner_slice = slice(0, num_inner)
        x_bnd = coords_bnd[inner_slice, 0:1]
        y_bnd = coords_bnd[inner_slice, 1:2]
        nx = x_bnd / R
        ny = y_bnd / R

        Tx = sig_xx[inner_slice] * nx + sig_xy[inner_slice] * ny
        Ty = sig_xy[inner_slice] * nx + sig_yy[inner_slice] * ny
        loss_bnd = loss_bnd + torch.mean(Tx**2 + Ty**2)

        # 2. Outer boundaries (second half)
        outer_slice = slice(num_inner, None)
        x_out = coords_bnd[outer_slice, 0:1]
        y_out = coords_bnd[outer_slice, 1:2]

        # ── Right boundary (x ≈ +L): σ_xx = T(y), σ_xy = 0 ──
        right_mask = (x_out > L - 1e-3).squeeze()
        if torch.any(right_mask):
            # Interpolate T(y) to the y-coords of right-edge boundary points
            y_right = y_out[right_mask]
            T_y = _interpolate_load_to_y(loads[i], sensor_y, y_right)
            loss_bnd = loss_bnd + torch.mean(
                (sig_xx[outer_slice][right_mask] - T_y)**2
            )
            loss_bnd = loss_bnd + torch.mean(
                sig_xy[outer_slice][right_mask]**2
            )

        # ── Left boundary (x ≈ −L): σ_xx = T(y), σ_xy = 0 ──
        #    (matching traction for finite-plate approximation)
        left_mask = (x_out < -L + 1e-3).squeeze()
        if torch.any(left_mask):
            y_left = y_out[left_mask]
            T_y_left = _interpolate_load_to_y(loads[i], sensor_y, y_left)
            loss_bnd = loss_bnd + torch.mean(
                (sig_xx[outer_slice][left_mask] - T_y_left)**2
            )
            loss_bnd = loss_bnd + torch.mean(
                sig_xy[outer_slice][left_mask]**2
            )

        # ── Top boundary (y ≈ +L): σ_yy = 0, σ_xy = 0 ──
        top_mask = (y_out > L - 1e-3).squeeze()
        if torch.any(top_mask):
            loss_bnd = loss_bnd + torch.mean(
                sig_yy[outer_slice][top_mask]**2
            )
            loss_bnd = loss_bnd + torch.mean(
                sig_xy[outer_slice][top_mask]**2
            )

        # ── Bottom boundary (y ≈ −L): σ_yy = 0, σ_xy = 0 ──
        bottom_mask = (y_out < -L + 1e-3).squeeze()
        if torch.any(bottom_mask):
            loss_bnd = loss_bnd + torch.mean(
                sig_yy[outer_slice][bottom_mask]**2
            )
            loss_bnd = loss_bnd + torch.mean(
                sig_xy[outer_slice][bottom_mask]**2
            )

    loss_pde = loss_pde / num_loads
    loss_bnd = loss_bnd / num_loads

    return loss_pde, loss_bnd


def main():
    parser = argparse.ArgumentParser(description="PI-DeepONet training with RAR")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to config file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced epochs & loads for testing")
    args = parser.parse_args()

    # ─── Resolve config path ───
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(_project_root, config_path)
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(config_path)
    exp_name = config['experiment']['name']
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  {config['experiment']['desc']}")
    print(f"{'='*60}")

    # ─── Set seeds ───
    torch.manual_seed(42)
    np.random.seed(42)

    # ─── Device ───
    device = get_device(config)
    print(f"  Device: {device}")

    # ─── Problem parameters ───
    L = 1.0   # Plate half-width
    R = 0.2   # Hole radius
    num_sensors = config['model']['branch_layers'][0]

    # ─── Quick mode overrides ───
    if args.quick:
        config['training']['epochs'] = min(config['training']['epochs'], 2000)
        config['training']['num_loads'] = min(config['training']['num_loads'], 20)
        config['training']['num_domain'] = min(config['training']['num_domain'], 500)
        config['training']['num_boundary'] = min(config['training']['num_boundary'], 100)
        if config['rar'].get('enabled', False):
            config['rar']['frequency'] = min(config['rar'].get('frequency', 500), 500)
        print("  [QUICK] Quick mode enabled (reduced scale)")

    # ─── Initialize GRF ───
    grf = GaussianRandomField(
        num_sensors=num_sensors,
        length_scale=0.2,
        variance=1.0,
        domain=(-L, L),
    )
    # Sensor y-coordinates (sorted) for interpolating T(y) on boundaries
    sensor_y = torch.tensor(grf.sensors, dtype=torch.float32, device=device)

    # ─── Initialize model ───
    model = PIDeepONet(
        branch_layers=config['model']['branch_layers'],
        trunk_layers=config['model']['trunk_layers'],
        num_outputs=2,
    ).to(device)

    print(f"  Model parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # ─── Generate initial training data ───
    num_loads = config['training']['num_loads']
    num_domain = config['training']['num_domain']
    num_boundary = config['training']['num_boundary']

    loads = torch.tensor(
        grf.sample(num_loads), dtype=torch.float32, device=device
    )
    coords_dom = sample_domain_points(num_domain, L, R, device=device)
    coords_bnd = sample_boundary_points(num_boundary, L, R, device=device)

    print(f"  Initial: {num_loads} loads, {num_domain} domain pts, "
          f"{num_boundary} boundary pts")

    # ─── Setup logging ───
    if os.path.exists('/kaggle/working'):
        results_dir = os.path.join('/kaggle/working', 'results')
    else:
        results_dir = os.path.join(_project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = LossLogger(os.path.join(results_dir, f"{exp_name}_losses.csv"))

    # ─── Training parameters ───
    epochs = config['training']['epochs']
    rar_enabled = config['rar'].get('enabled', False)
    log_freq = config['training'].get('log_frequency', 500)
    save_freq = config['training'].get('save_frequency', 2000)
    bc_weight = config['training'].get('bc_weight', 10.0)

    print(f"  Training for {epochs} epochs...")
    print(f"  RAR: {'enabled (' + config['rar'].get('type', 'none') + ')' if rar_enabled else 'disabled'}")
    print()

    # ═══════════════════════════════════════════════════════════
    #  Training Loop
    # ═══════════════════════════════════════════════════════════

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss_pde, loss_bnd = compute_losses(
            model, loads, coords_dom, coords_bnd, sensor_y, R, L
        )

        total_loss = loss_pde + bc_weight * loss_bnd
        total_loss.backward()
        optimizer.step()

        # ─── Logging ───
        if epoch % log_freq == 0 or epoch == 1:
            pde_val = loss_pde.item()
            bnd_val = loss_bnd.item()
            total_val = total_loss.item()
            print(f"  Epoch {epoch:6d} | Loss: {total_val:.4e} | "
                  f"PDE: {pde_val:.4e} | BC: {bnd_val:.4e} | "
                  f"Pts: {coords_dom.shape[0]} | Loads: {loads.shape[0]}")
            logger.log(epoch, total_val, pde_val, bnd_val,
                       coords_dom.shape[0], loads.shape[0])

        # ─── Save best model ───
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(),
                       os.path.join(results_dir, f"{exp_name}_best.pt"))

        # ─── Checkpoint ───
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, os.path.join(results_dir, f"{exp_name}_checkpoint_ep{epoch}.pt"))

        # ─── RAR refinement step ───
        if rar_enabled and epoch % config['rar']['frequency'] == 0:
            rar_type = config['rar']['type']
            print(f"\n  --- RAR Refinement at Epoch {epoch} ({rar_type}) ---")

            if rar_type in ["collocation", "combined"]:
                coords_dom = rar_add_collocation_points(
                    model, loads, coords_dom, config, L, R, device
                )
                print(f"  Added collocation points -> {coords_dom.shape[0]} total")

            if rar_type in ["load", "combined"]:
                loads = rar_add_load_functions(
                    model, loads, coords_dom, grf, config, device
                )
                print(f"  Added load functions -> {loads.shape[0]} total")

            print()

    # ─── Save final model ───
    save_path = os.path.join(results_dir, f"{exp_name}_final.pt")
    torch.save(model.state_dict(), save_path)
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Training complete: {exp_name}")
    print(f"  Best loss: {best_loss:.4e}")
    print(f"  Model saved: {save_path}")
    print(f"  Losses CSV: results/{exp_name}_losses.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
