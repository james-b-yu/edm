import torch
import numpy as np
from collections import deque
from configs.dataset_reg_config import get_dataset_info
from configs.bond_config import TENSOR_1

def bfs(start, A):
    queue = deque()
    queue.append(start)
    reachable = set()

    while (len(queue) > 0):
        v = queue.popleft()
        reachable.add(v)
        for (u, edge) in enumerate(A[v]):
            if edge and u not in reachable:
                queue.append(u)

    return list(reachable)


def get_graph_components(A):
    v = 0
    visited = set()
    components = []

    for v in range(A.shape[0]):
        if v not in visited:
            reachable = bfs(v, A)
            visited.update(reachable)
            components.append(reachable)

    return components
    

def get_adjacency(coords, atom_types, dataset_info):
    num_atoms = coords.shape[0]
    
    # Compute pairwise distances using broadcasting
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
    
    # Get atom type mappings
    atom_1_types = atom_types.unsqueeze(0).expand(num_atoms, num_atoms)
    atom_2_types = atom_types.unsqueeze(1).expand(num_atoms, num_atoms)

    thresholds = TENSOR_1[atom_1_types, atom_2_types]
    distance_diffs = distances - thresholds
    distance_diffs[distance_diffs < 0] = 0
    A = (distance_diffs == 0)
    
    return A, distance_diffs


def get_disconnection_penalty(coords, features, mol_sizes, time_fracs, dataset_name='qm9', use_h=True):
    dataset_info = get_dataset_info(dataset_name, use_h)
    num_types = len(dataset_info['atom_types'])

    coords = coords.view(-1, 3)
    features = features[:, :num_types].view(-1, num_types).type(torch.float32)
    atom_types = torch.argmax(features, dim=1)
    penalties = torch.zeros((len(mol_sizes),))

    idx = 0
    for i, size in enumerate(mol_sizes):
        penalty = 0
        mol_coords = coords[idx : idx + size]
        mol_types = atom_types[idx : idx + size]
        time_frac = time_fracs[idx : idx + size][0]

        A, distances = get_adjacency(mol_coords, mol_types, dataset_info)
        components = get_graph_components(A)
        if len(components) > 1:
            for component in components:
                dist_to_other = distances[component]
                dist_to_other[:, component] = np.inf
                argmin_flat = torch.argmin(dist_to_other)
                v = argmin_flat // dist_to_other.shape[1]
                u = argmin_flat % dist_to_other.shape[1]
                penalty += dist_to_other[v, u]
        

        penalties[i] = penalty / size * (1 - time_frac) # Penalty weighted more earlier in the noising process
        idx += size

    return penalties