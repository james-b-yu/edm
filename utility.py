import torch

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



def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    # if float(grad_norm) > max_grad_norm:
    #     print(f'Clipped gradient with value {grad_norm:.1f} '
    #           f'while allowed {max_grad_norm:.1f}')
    return grad_norm


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
