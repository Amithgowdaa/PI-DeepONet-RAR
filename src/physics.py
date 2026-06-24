import torch

def compute_pde_residuals(coords, u, v, E=200e9, nu=0.3, plane_stress=True):
    """
    Computes 2D linear elasticity Navier-Cauchy PDE residuals.
    
    Args:
        coords: (N, 2) Tensor representing (x, y) coordinates.
        u: (N, 1) Tensor representing displacement in x-direction.
        v: (N, 1) Tensor representing displacement in y-direction.
        E: Young's Modulus.
        nu: Poisson's ratio.
        plane_stress: If True, uses plane stress equations. If False, plane strain.
        
    Returns:
        res_x: (N, 1) Tensor of PDE residual in x-direction.
        res_y: (N, 1) Tensor of PDE residual in y-direction.
        stresses: Dictionary containing stresses: 'sigma_xx', 'sigma_yy', 'sigma_xy'.
    """
    # Compute Lame parameters
    if plane_stress:
        lmbda = E * nu / ((1.0 + nu) * (1.0 - nu))
        mu = E / (2.0 * (1.0 + nu))
    else: # plane strain
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        
    # Gradients of u
    u_g = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = u_g[:, 0:1]
    u_y = u_g[:, 1:2]
    
    # Gradients of v
    v_g = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_x = v_g[:, 0:1]
    v_y = v_g[:, 1:2]
    
    # Strains
    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)
    
    # Stresses (Hooke's Law)
    if plane_stress:
        factor = E / (1.0 - nu**2)
        sigma_xx = factor * (eps_xx + nu * eps_yy)
        sigma_yy = factor * (eps_yy + nu * eps_xx)
        sigma_xy = (E / (1.0 + nu)) * eps_xy
    else:
        # Plane strain
        trace_eps = eps_xx + eps_yy
        sigma_xx = lmbda * trace_eps + 2.0 * mu * eps_xx
        sigma_yy = lmbda * trace_eps + 2.0 * mu * eps_yy
        sigma_xy = 2.0 * mu * eps_xy
        
    # Gradients of stresses for equilibrium:
    # d(sigma_xx)/dx + d(sigma_xy)/dy = 0
    # d(sigma_xy)/dx + d(sigma_yy)/dy = 0
    
    sigma_xx_g = torch.autograd.grad(sigma_xx, coords, grad_outputs=torch.ones_like(sigma_xx), create_graph=True, retain_graph=True)[0]
    sigma_xx_x = sigma_xx_g[:, 0:1]
    
    sigma_xy_g = torch.autograd.grad(sigma_xy, coords, grad_outputs=torch.ones_like(sigma_xy), create_graph=True, retain_graph=True)[0]
    sigma_xy_x = sigma_xy_g[:, 0:1]
    sigma_xy_y = sigma_xy_g[:, 1:2]
    
    sigma_yy_g = torch.autograd.grad(sigma_yy, coords, grad_outputs=torch.ones_like(sigma_yy), create_graph=True, retain_graph=True)[0]
    sigma_yy_y = sigma_yy_g[:, 1:2]
    
    # Residuals
    res_x = sigma_xx_x + sigma_xy_y
    res_y = sigma_xy_x + sigma_yy_y
    
    stresses = {
        'sigma_xx': sigma_xx,
        'sigma_yy': sigma_yy,
        'sigma_xy': sigma_xy
    }
    
    return res_x, res_y, stresses
