import torch

def collate_fn(batch):
    """ Custom collate function to process batches of EDMDatasetItem. """
    
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

    # Fix: Correctly construct `reduce` to map nodes to edges (N, NN)
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

    demean = torch.block_diag(*demean_list)  # ✅ Ensures correct batching

    return {
        "n_nodes": n_nodes,
        "coords": coords,
        "features": features,
        "edges": edges,   # ✅ Dynamically computed
        "reduce": reduce,  # ✅ Corrected shape (N, NN)
        "demean": demean  # ✅ Included for centering
    }

