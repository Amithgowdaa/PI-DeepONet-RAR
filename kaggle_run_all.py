"""
Kaggle / Colab Runner — PI-DeepONet-RAR Full Experiment
========================================================

Runs all 4 experimental arms sequentially and generates comparison plots.

Usage (from project root):
    python kaggle_run_all.py              # Full run
    python kaggle_run_all.py --quick      # Quick test run (~5 min)
    python kaggle_run_all.py --arms baseline rar_collocation  # Specific arms

In Kaggle notebook:
    !python kaggle_run_all.py --quick
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import torch
from model import PIDeepONet
from analytical_kirsch import (
        analytical_kirsch_stress,
        stress_concentration_factor,
    )
from physics import compute_stresses

# ─── Setup paths ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'plots')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fem_baseline'))

ALL_ARMS = ['baseline', 'rar_collocation', 'rar_load', 'rar_combined']
ARM_COLORS = {
    'baseline': '#6C757D',
    'rar_collocation': '#0D6EFD',
    'rar_load': '#198754',
    'rar_combined': '#DC3545',
}
ARM_LABELS = {
    'baseline': 'Baseline (Uniform)',
    'rar_collocation': 'Collocation RAR',
    'rar_load': 'Load RAR',
    'rar_combined': 'Combined RAR',
}


def run_experiment(arm_name, quick=False):
    """Run a single experimental arm via train.py."""
    config_path = os.path.join(PROJECT_ROOT, 'configs', f'{arm_name}.yaml')
    cmd = [sys.executable, os.path.join(SRC_DIR, 'train.py'),
           '--config', config_path]
    if quick:
        cmd.append('--quick')

    print(f"\n{'='*60}")
    print(f"  Running: {ARM_LABELS[arm_name]}")
    print(f"  Config:  {config_path}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ⚠ {arm_name} returned non-zero exit code: {result.returncode}")
    else:
        print(f"  ✓ {arm_name} completed in {elapsed:.1f}s")

    return result.returncode


def plot_loss_comparison(arms):
    """Plot training loss curves for all completed arms."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PI-DeepONet-RAR: Training Loss Comparison', fontsize=14, fontweight='bold')

    loss_types = [
        ('total_loss', 'Total Loss'),
        ('pde_loss', 'PDE Residual Loss'),
        ('bc_loss', 'Boundary Condition Loss'),
    ]

    for ax, (col, title) in zip(axes, loss_types):
        for arm in arms:
            csv_path = os.path.join(RESULTS_DIR, f'{arm}_losses.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            ax.semilogy(df['epoch'], df[col],
                        label=ARM_LABELS[arm],
                        color=ARM_COLORS[arm],
                        alpha=0.85,
                        linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'loss_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Loss comparison plot saved: {save_path}")


def plot_adaptive_growth(arms):
    """Plot how domain points and loads grow over training for RAR arms."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RAR: Adaptive Growth of Training Data', fontsize=14, fontweight='bold')

    for arm in arms:
        csv_path = os.path.join(RESULTS_DIR, f'{arm}_losses.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        ax1.plot(df['epoch'], df['num_domain_pts'],
                 label=ARM_LABELS[arm],
                 color=ARM_COLORS[arm],
                 linewidth=1.5)
        ax2.plot(df['epoch'], df['num_loads'],
                 label=ARM_LABELS[arm],
                 color=ARM_COLORS[arm],
                 linewidth=1.5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Domain Collocation Points')
    ax1.set_title('Collocation Point Growth')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Active Load Functions')
    ax2.set_title('Load Function Growth')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'rar_growth.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 RAR growth plot saved: {save_path}")


def compute_validation_errors(arms):
    """
    Evaluate each trained model against the analytical Kirsch solution
    and report L2 relative errors and SCF accuracy.
    """


    # Validation grid (outside the hole)
    R = 0.2
    L = 1.0
    T = 1.0

    # Generate validation points
    np.random.seed(999)
    val_points = []
    while len(val_points) < 2000:
        pts = np.random.uniform(-L, L, (4000, 2))
        valid = pts[np.sum(pts**2, axis=1) >= R**2]
        val_points.extend(valid.tolist())
    val_points = np.array(val_points[:2000])

    # Analytical stresses
    sxx_true, syy_true, sxy_true = analytical_kirsch_stress(
        val_points[:, 0], val_points[:, 1], R=R, T=T
    )

    # Read num_sensors from the first arm's config
    first_config_path = os.path.join(PROJECT_ROOT, 'configs', f'{arms[0]}.yaml')
    with open(first_config_path, 'r') as f:
        first_config = yaml.safe_load(f)
    num_sensors = first_config['model']['branch_layers'][0]

    # Uniform tension load for evaluation
    load_uniform = torch.ones(1, num_sensors, dtype=torch.float32) * T

    results = {}
    true_scf = stress_concentration_factor(R, T)

    print(f"\n{'='*60}")
    print(f"  Validation Against Analytical Kirsch Solution")
    print(f"  True SCF = {true_scf:.4f}")
    print(f"{'='*60}")

    for arm in arms:
        model_path = os.path.join(RESULTS_DIR, f'{arm}_best.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(RESULTS_DIR, f'{arm}_final.pt')
        if not os.path.exists(model_path):
            print(f"  ⚠ No model found for {arm}")
            continue

        # Load architecture from the arm's config file
        config_path = os.path.join(PROJECT_ROOT, 'configs', f'{arm}.yaml')
        with open(config_path, 'r') as f:
            arm_config = yaml.safe_load(f)
        branch_layers = arm_config['model']['branch_layers']
        trunk_layers = arm_config['model']['trunk_layers']

        # Load model
        model = PIDeepONet(
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
            num_outputs=2,
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu',
                                         weights_only=True))
        model.eval()

        # Predict stresses using autograd (need gradients on coords)
        coords_val = torch.tensor(val_points, dtype=torch.float32, requires_grad=True)
        pred = model(load_uniform, coords_val)
        u_pred = pred[0, :, 0:1]
        v_pred = pred[0, :, 1:2]
        stresses = compute_stresses(coords_val, u_pred, v_pred)

        sxx_pred = stresses['sigma_xx'].detach().numpy().squeeze()
        syy_pred = stresses['sigma_yy'].detach().numpy().squeeze()
        sxy_pred = stresses['sigma_xy'].detach().numpy().squeeze()

        # L2 relative errors
        l2_sxx = np.linalg.norm(sxx_pred - sxx_true) / (np.linalg.norm(sxx_true) + 1e-10)
        l2_syy = np.linalg.norm(syy_pred - syy_true) / (np.linalg.norm(syy_true) + 1e-10)
        l2_sxy = np.linalg.norm(sxy_pred - sxy_true) / (np.linalg.norm(sxy_true) + 1e-10)

        # SCF: evaluate at (0, R)
        scf_coord = torch.tensor([[0.0, R]], dtype=torch.float32, requires_grad=True)
        pred_scf = model(load_uniform, scf_coord)
        u_scf = pred_scf[0, :, 0:1]
        v_scf = pred_scf[0, :, 1:2]
        stress_scf = compute_stresses(scf_coord, u_scf, v_scf)
        predicted_scf = stress_scf['sigma_xx'].item() / T

        results[arm] = {
            'l2_sxx': l2_sxx,
            'l2_syy': l2_syy,
            'l2_sxy': l2_sxy,
            'scf': predicted_scf,
        }

        print(f"\n  {ARM_LABELS[arm]}:")
        print(f"    L2 errors: σ_xx={l2_sxx:.4f}, σ_yy={l2_syy:.4f}, σ_xy={l2_sxy:.4f}")
        print(f"    SCF: {predicted_scf:.4f} (true: {true_scf:.4f}, "
              f"error: {abs(predicted_scf - true_scf):.4f})")

    # Save results table
    if results:
        df = pd.DataFrame(results).T
        df.index.name = 'arm'
        csv_path = os.path.join(RESULTS_DIR, 'validation_results.csv')
        df.to_csv(csv_path)
        print(f"\n  📊 Validation results saved: {csv_path}")

        # Plot SCF comparison
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        arm_names = list(results.keys())
        scf_vals = [results[a]['scf'] for a in arm_names]
        colors = [ARM_COLORS[a] for a in arm_names]
        labels = [ARM_LABELS[a] for a in arm_names]

        bars = ax.bar(labels, scf_vals, color=colors, edgecolor='white', linewidth=1.5)
        ax.axhline(y=true_scf, color='black', linestyle='--', linewidth=2,
                    label=f'Theoretical ({true_scf:.1f})')
        ax.set_ylabel('Stress Concentration Factor')
        ax.set_title('SCF Prediction Accuracy by Arm')
        ax.legend()
        ax.set_ylim(0, 4.0)
        ax.grid(axis='y', alpha=0.3)

        save_path = os.path.join(PLOTS_DIR, 'scf_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 SCF comparison plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PI-DeepONet-RAR: Run All Experiments")
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode for testing (fewer epochs)')
    parser.add_argument('--arms', nargs='+', default=ALL_ARMS,
                        choices=ALL_ARMS,
                        help='Which experimental arms to run')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only generate plots')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "═" * 60)
    print("  PI-DeepONet-RAR: Full Experiment Suite")
    print("  Arms: " + ", ".join(args.arms))
    print("  Mode: " + ("Quick" if args.quick else "Full"))
    print("═" * 60)

    # ─── Run experiments ───
    if not args.skip_training:
        overall_start = time.time()
        for arm in args.arms:
            run_experiment(arm, quick=args.quick)
        total_time = time.time() - overall_start
        print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ─── Generate comparison plots ───
    print("\n  Generating comparison plots...")
    plot_loss_comparison(args.arms)
    plot_adaptive_growth(args.arms)

    # ─── Validation against analytical solution ───
    print("\n  Running validation...")
    compute_validation_errors(args.arms)

    print("\n" + "═" * 60)
    print("  All done! Check results/ directory for outputs.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
