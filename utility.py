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
    # grad_norm = torch.nn.utils.clip_grad_norm_(
    #     model.parameters(), max_norm=max_grad_norm.item(), norm_type=2.0
    # )

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"[ERROR] NaN detected in gradients of {name}")



    # Ensure grad_norm is also on the correct device
    grad_norm_tensor = torch.tensor([grad_norm], device=device)

    # Update gradnorm_queue with latest gradient norm
    gradnorm_queue = torch.cat([gradnorm_queue, grad_norm_tensor])

    # Keep queue size reasonable (e.g., last 100 values)
    if gradnorm_queue.numel() > 100:
        gradnorm_queue = gradnorm_queue[-100:]

    return grad_norm, gradnorm_queue 





# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()