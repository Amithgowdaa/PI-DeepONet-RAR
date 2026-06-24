import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np

from model import PIDeepONet
from physics import compute_pde_residuals
from data_generation import GaussianRandomField

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def sample_domain_points(num_points, L=1.0, R=0.2):
    """Samples points inside the plate with a circular hole of radius R."""
    points = []
    while len(points) < num_points:
        # Uniform sampling in [-L, L] x [-L, L]
        pts = np.random.uniform(-L, L, (num_points * 2, 2))
        # Keep points outside the hole
        valid = pts[np.sum(pts**2, axis=1) >= R**2]
        points.extend(valid)
    return torch.tensor(np.array(points[:num_points]), dtype=torch.float32, requires_grad=True)

def sample_boundary_points(num_points, L=1.0, R=0.2):
    """Samples points along the boundaries: inner hole and outer boundaries."""
    points = []
    
    # 1. Inner circle boundary (traction free)
    num_inner = num_points // 2
    theta = np.random.uniform(0, 2 * np.pi, num_inner)
    inner_pts = np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
    
    # 2. Outer boundaries: x = +L (tension), x = -L, y = +L, y = -L
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
    return torch.tensor(points, dtype=torch.float32, requires_grad=True)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="PI-DeepONet training with RAR")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Check if config exists, if not construct absolute or relative path
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        
    config = load_config(config_path)
    print(f"Loaded config: {config['experiment']['name']}")
    print(config['experiment']['desc'])
    
    # Problem dimensions
    L = 1.0
    R = 0.2
    num_sensors = config['model']['branch_layers'][0]
    
    # Initialize Gaussian Random Field for boundary load functions (tension T(y) at x = L)
    grf = GaussianRandomField(num_sensors=num_sensors, length_scale=0.2, variance=1.0, domain=(-L, L))
    
    # Initialize Model
    # Branch output dim: trunk_output_dim * 2 (since u, v outputs)
    model = PIDeepONet(
        branch_layers=config['model']['branch_layers'],
        trunk_layers=config['model']['trunk_layers'],
        num_outputs=2
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Generate initial datasets
    num_loads = config['training']['num_loads']
    num_domain = config['training']['num_domain']
    num_boundary = config['training']['num_boundary']
    
    # Branch inputs: load functions
    loads = torch.tensor(grf.sample(num_loads), dtype=torch.float32) # (num_loads, num_sensors)
    
    # Trunk inputs: domain & boundary coordinates
    coords_dom = sample_domain_points(num_domain, L, R)
    coords_bnd = sample_boundary_points(num_boundary, L, R)
    
    print(f"Initial setup: {num_loads} loads, {num_domain} domain points, {num_boundary} boundary points")
    
    # Training Loop
    epochs = config['training']['epochs']
    rar_enabled = config['rar']['enabled']
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # We will train in paired mode for physics equations or Cartesian product.
        # Here we perform Cartesian product or batching logic.
        # Let's do a sub-batch of loads to fit memory
        outputs_dom = model(loads, coords_dom) # (N_loads, N_dom, 2)
        outputs_bnd = model(loads, coords_bnd) # (N_loads, N_bnd, 2)
        
        loss_pde = 0.0
        # Compute PDE residual loss over all loads
        for i in range(num_loads):
            u_dom = outputs_dom[i, :, 0:1]
            v_dom = outputs_dom[i, :, 1:2]
            res_x, res_y, stresses = compute_pde_residuals(coords_dom, u_dom, v_dom)
            loss_pde += torch.mean(res_x**2 + res_y**2)
            
        loss_pde = loss_pde / num_loads
        
        loss_bnd = 0.0
        # Compute boundary loss
        # Outer boundary conditions and inner traction-free boundary conditions
        for i in range(num_loads):
            u_bnd = outputs_bnd[i, :, 0:1]
            v_bnd = outputs_bnd[i, :, 1:2]
            
            # Boundary stresses
            _, _, stresses_bnd = compute_pde_residuals(coords_bnd, u_bnd, v_bnd)
            sig_xx = stresses_bnd['sigma_xx']
            sig_yy = stresses_bnd['sigma_yy']
            sig_xy = stresses_bnd['sigma_xy']
            
            # 1. Inner boundary (first half of bnd points) traction-free: T_i = sigma_ij * n_j = 0
            # Normal vector on circle: n = (x/R, y/R)
            inner_slice = slice(0, num_boundary // 2)
            x_bnd = coords_bnd[inner_slice, 0:1]
            y_bnd = coords_bnd[inner_slice, 1:2]
            nx = x_bnd / R
            ny = y_bnd / R
            
            Tx = sig_xx[inner_slice] * nx + sig_xy[inner_slice] * ny
            Ty = sig_xy[inner_slice] * nx + sig_yy[inner_slice] * ny
            loss_bnd += torch.mean(Tx**2 + Ty**2)
            
            # 2. Outer boundary tension at x = L: sigma_xx = T(y), sigma_xy = 0
            # Tension is given by the loading function loads[i] interpolated at boundary y-coordinates
            # For simplicity, we define a basic BC loss on outer bounds
            # x = -L (fixed displacement or symmetric), y = +/- L (free)
            # Let's add simple BC residuals
            outer_slice = slice(num_boundary // 2, None)
            x_out = coords_bnd[outer_slice, 0:1]
            # Uniaxial tension boundary at x = L
            right_mask = (x_out > L - 1e-3).squeeze()
            if torch.any(right_mask):
                loss_bnd += torch.mean(sig_xy[outer_slice][right_mask]**2)
                # target tension from branch net input (interpolated or average load value)
                target_tension = torch.mean(loads[i]) # proxy for this example
                loss_bnd += torch.mean((sig_xx[outer_slice][right_mask] - target_tension)**2)
                
        loss_bnd = loss_bnd / num_loads
        
        # Total loss
        loss = loss_pde + 10.0 * loss_bnd
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | Total Loss: {loss.item():.4e} | PDE Loss: {loss_pde.item():.4e} | BC Loss: {loss_bnd.item():.4e}")
            
        # RAR refinement step
        if rar_enabled and epoch % config['rar']['frequency'] == 0:
            print(f"--- Running RAR Refinement at Epoch {epoch} ---")
            rar_type = config['rar']['type']
            
            if rar_type in ["collocation", "combined"]:
                # Evaluate residuals on candidate coordinates and add those with highest residual
                coords_candidate = sample_domain_points(2000, L, R)
                candidate_residuals = []
                with torch.no_grad():
                    # Evaluate for first load
                    outputs_candidate = model(loads[0:1], coords_candidate)
                    u_cand = outputs_candidate[0, :, 0:1].clone().detach().requires_grad_(True)
                    v_cand = outputs_candidate[0, :, 1:2].clone().detach().requires_grad_(True)
                    rx, ry, _ = compute_pde_residuals(coords_candidate, u_cand, v_cand)
                    res_mag = (rx**2 + ry**2).squeeze().cpu().numpy()
                    
                # Find indices of top points
                top_indices = np.argsort(res_mag)[-config['rar']['num_points_to_add']:]
                new_points = coords_candidate[top_indices].detach()
                coords_dom = torch.cat([coords_dom.detach(), new_points], dim=0).requires_grad_(True)
                num_domain = coords_dom.shape[0]
                print(f"Added {len(top_indices)} new collocation points. Total domain points: {num_domain}")
                
            if rar_type in ["load", "combined"]:
                # Generate new candidate loads, evaluate prediction difficulty, and add top loads
                new_loads = torch.tensor(grf.sample(50), dtype=torch.float32)
                # Compute average residual on domain coordinates
                load_residuals = []
                for j in range(50):
                    with torch.no_grad():
                        outputs_cand_load = model(new_loads[j:j+1], coords_dom)
                        u_l = outputs_cand_load[0, :, 0:1].clone().detach().requires_grad_(True)
                        v_l = outputs_cand_load[0, :, 1:2].clone().detach().requires_grad_(True)
                        rx, ry, _ = compute_pde_residuals(coords_dom, u_l, v_l)
                        mean_res = torch.mean(rx**2 + ry**2).item()
                        load_residuals.append(mean_res)
                        
                top_load_indices = np.argsort(load_residuals)[-config['rar']['num_loads_to_add']:]
                new_active_loads = new_loads[top_load_indices]
                loads = torch.cat([loads, new_active_loads], dim=0)
                num_loads = loads.shape[0]
                print(f"Added {len(top_load_indices)} new load functions. Total active loads: {num_loads}")
                
    # Save trained model structure
    os.makedirs("results", exist_ok=True)
    save_path = f"results/{config['experiment']['name']}_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Finished training. Model saved to {save_path}")

if __name__ == "__main__":
    main()
