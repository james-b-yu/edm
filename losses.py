import torch
import torch.nn.functional as F


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    """
    Ensures that masked values are correctly ignored.
    """
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def compute_loss_and_nll(model, x, h, n_nodes, edges):
    """
    Computes the negative log-likelihood (NLL) loss for the diffusion model.

    Args:
        model (torch.nn.Module): The generative diffusion model.
        x (torch.Tensor): Molecular coordinates of shape (num_atoms, 3).
        h (torch.Tensor): Node features of shape (num_atoms, feature_dim).
        n_nodes (torch.Tensor): Number of atoms per molecule.
        edges (torch.Tensor): Edge connectivity indices.

    Returns:
        tuple: (Negative Log-Likelihood loss, regularization term, mean absolute latent variable)
    """

    device = x.device
    batch_size = n_nodes.shape[0]  # Number of molecules in batch

    # Ensure edge_mask is properly shaped
    edge_mask = torch.ones((x.shape[0], x.shape[0]), device=device)  # No explicit mask
    edge_mask = edge_mask.view(batch_size, -1)  # Reshape to batch size

    assert_correctly_masked(x, n_nodes.unsqueeze(-1))  # Ensure masking is correct

    # Compute NLL loss using the model
    nll = model(n_nodes, x, h, edges, None, None, None)  # Pass input through model

    # Compute regularization term (default = 0)
    reg_term = torch.tensor([0.0], device=device)

    # Compute mean absolute value of latent variables (optional logging metric)
    mean_abs_z = torch.mean(torch.abs(nll)).item()

    return nll.mean(0), reg_term, mean_abs_z


def compute_loss(model, batch, noise_schedule):
    """
    Computes the loss for the diffusion model by comparing predicted noise with actual noise.
    
    Args:
        model (torch.nn.Module): The generative diffusion model.
        batch (dict): Batch of molecular data.
        noise_schedule (dict): The noise schedule used for diffusion training.
    
    Returns:
        tuple: (Loss value, Negative log-likelihood, Regularization term)
    """
    device = batch["coords"].device
    num_steps = noise_schedule["alpha"].shape[0]  # Total diffusion steps

    # Sample random timestep for each molecule in the batch
    t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
    time_tensor = t / num_steps  # Normalize time

    # Expand time to match the number of atoms
    num_atoms_total = batch["coords"].shape[0]  
    time_expanded = time_tensor.repeat_interleave(batch["n_nodes"]).reshape(num_atoms_total, 1)

    # Retrieve noise scaling factors
    alpha_t = noise_schedule["alpha"][t].unsqueeze(-1)  # (batch_size, 1)
    sigma_t = noise_schedule["sigma"][t].unsqueeze(-1)  # (batch_size, 1)

    alpha_t = torch.clamp(alpha_t, min=1e-3, max=1.0)
    sigma_t = torch.clamp(sigma_t, min=1e-3, max=1.0)
    print(f"alpha_t: {alpha_t.mean().item()}, sigma_t: {sigma_t.mean().item()}")


    # Expand scaling factors for all atoms in the batch
    alpha_t_expanded = alpha_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)
    sigma_t_expanded = sigma_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)

    # Generate Gaussian noise
    noise_coords = torch.randn_like(batch["coords"], dtype=torch.float32)
    noise_features = torch.randn_like(batch["features"], dtype=torch.float32)

    # Apply diffusion process (adding noise)
    zt_coords = alpha_t_expanded * batch["coords"] + sigma_t_expanded * noise_coords
    zt_features = alpha_t_expanded * batch["features"] + sigma_t_expanded * noise_features

    # Model predicts the noise
    predicted_coords, predicted_features = model(
        batch["n_nodes"], zt_coords, zt_features, batch["edges"], 
        batch["reduce"], batch["demean"], time_expanded
    )

    # Compute MSE loss between predicted noise and actual noise
    loss_coords = F.mse_loss(predicted_coords, noise_coords, reduction='mean')
    loss_features = F.mse_loss(predicted_features, noise_features, reduction='mean')
    loss = loss_coords + loss_features  # Total loss

    return loss, loss_coords, loss_features
