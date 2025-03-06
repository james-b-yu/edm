import torch
import numpy as np

def collate_fn(batch):

    n_nodes = torch.tensor([item.n_nodes for item in batch], dtype=torch.int64)
    coords = torch.cat([item.coords for item in batch])
    features = torch.cat([item.features for item in batch])

    # Compute node offset for batch processing
    node_offset = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), n_nodes[:-1]]), dim=0)

    # Compute edges dynamically
    edges_list = []
    for i, item in enumerate(batch):
        num_atoms = item.n_nodes
        molecule_edges = torch.cartesian_prod(torch.arange(num_atoms), torch.arange(num_atoms)) + node_offset[i]
        edges_list.append(molecule_edges)
    
    edges = torch.cat(edges_list, dim=0)

    # construct `reduce` to map nodes to edges (N, NN)
    N = coords.shape[0]   # Total number of nodes
    NN = edges.shape[0]   # Total number of edges

    reduce = torch.zeros((N, NN), dtype=torch.float32)
    for i in range(NN):
        reduce[edges[i, 0], i] = 1.0  # Mark where edges originate

    # Compute demean matrix (ensures zero center of mass for molecules)
    demean_list = []
    for item in batch:
        N_i = item.n_nodes
        demean = torch.eye(N_i) - (torch.ones((N_i, N_i)) / N_i)
        demean_list.append(demean)

    demean = torch.block_diag(*demean_list)  

    return {
        "n_nodes": n_nodes,
        "coords": coords,
        "features": features,
        "edges": edges,   
        "reduce": reduce,
        "demean": demean  
    }

import torch

def gradient_clipping(model, gradnorm_queue, default_clip=1.0):
    """
    Clips gradients based on recent gradient norms to prevent instability.
    
    Args:
        model (torch.nn.Module): The neural network model.
        gradnorm_queue (torch.Tensor): Stores recent gradient norms for adaptive clipping.
        default_clip (float): Default gradient clipping threshold if history is unavailable.

    Returns:
        float: The final computed gradient norm.
    """

    device = gradnorm_queue.device  # Ensure new tensors match device

    # Prevent std() issue by ensuring at least 2 values exist
    if gradnorm_queue.numel() < 2:
        max_grad_norm = torch.tensor(default_clip, device=device)  # Use default if not enough history
    else:
        mean_grad = gradnorm_queue.mean().item()
        std_grad = gradnorm_queue.std().item()
        max_grad_norm = torch.tensor(1.5 * mean_grad + 2 * std_grad, device=device)

    # Clip gradients and return the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm.item(), norm_type=2.0
    )

    # Ensure grad_norm is also on the correct device
    grad_norm_tensor = torch.tensor([grad_norm], device=device)

    # Update gradnorm_queue with latest gradient norm
    gradnorm_queue = torch.cat([gradnorm_queue, grad_norm_tensor])

    # Keep queue size reasonable (e.g., last 100 values)
    if gradnorm_queue.numel() > 100:
        gradnorm_queue = gradnorm_queue[-100:]

    return grad_norm, gradnorm_queue 





# Rotation data augmntation
def random_rotation(x, n_nodes):
    """
    Apply a random 3D rotation to molecular coordinates.
    
    Args:
        x (torch.Tensor): Coordinates tensor of shape (num_atoms, 3).
        n_nodes (torch.Tensor): Number of nodes (atoms) per molecule.

    Returns:
        torch.Tensor: Rotated coordinates with the same shape as input.
    """
    device = x.device
    angle_range = np.pi * 2
    rotated_coords = []

    # Iterate over molecules
    split_coords = torch.split(x, n_nodes.tolist())  # Split coordinates per molecule
    for molecule_coords in split_coords:
        num_atoms = molecule_coords.shape[0]  # Number of atoms in the molecule
        
        if num_atoms == 0:  # Skip empty molecules (unlikely, but for safety)
            rotated_coords.append(molecule_coords)
            continue

        molecule_coords = molecule_coords.unsqueeze(0)  # Shape: (1, num_atoms, 3)

        # Build Rx
        Rx = torch.eye(3, device=device).unsqueeze(0)
        theta = torch.rand(1, 1, device=device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1, 1] = cos
        Rx[:, 1, 2] = sin
        Rx[:, 2, 1] = -sin
        Rx[:, 2, 2] = cos

        # Build Ry
        Ry = torch.eye(3, device=device).unsqueeze(0)
        theta = torch.rand(1, 1, device=device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0, 0] = cos
        Ry[:, 0, 2] = -sin
        Ry[:, 2, 0] = sin
        Ry[:, 2, 2] = cos

        # Build Rz
        Rz = torch.eye(3, device=device).unsqueeze(0)
        theta = torch.rand(1, 1, device=device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0, 0] = cos
        Rz[:, 0, 1] = sin
        Rz[:, 1, 0] = -sin
        Rz[:, 1, 1] = cos

        # Apply rotations
        molecule_coords = molecule_coords.transpose(1, 2)  # Shape: (1, 3, num_atoms)
        molecule_coords = torch.matmul(Rx, molecule_coords)
        molecule_coords = torch.matmul(Ry, molecule_coords)
        molecule_coords = torch.matmul(Rz, molecule_coords)
        molecule_coords = molecule_coords.transpose(1, 2).squeeze(0)  # Back to (num_atoms, 3)

        rotated_coords.append(molecule_coords)

    # Recombine rotated molecules
    return torch.cat(rotated_coords, dim=0).contiguous()
