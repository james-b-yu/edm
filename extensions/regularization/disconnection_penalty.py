import torch
import numpy as np
from collections import deque
from configs.dataset_reg_config import get_dataset_info
from configs.bond_config import BONDS_1, MARGIN_1

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
    
    
def get_distance_to_connection(atom_1, atom_2, distance):
    if atom_1 not in BONDS_1 or atom_2 not in BONDS_1[atom_1]:
        return np.inf
        
    threshold = (BONDS_1[atom_1][atom_2] + MARGIN_1) / 100
    return np.max(distance - threshold, 0)


def get_adjacency(coords, atom_types, dataset_info):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    A = torch.zeros((len(x), len(x)))
    distances = torch.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = torch.Tensor([x[i], y[i], z[i]])
            p2 = torch.Tensor([x[j], y[j], z[j]])
            distance = torch.sqrt(torch.sum((p1 - p2) ** 2)).item()
            
            atom_types_srt = sorted((atom_types[i], atom_types[j]))
            atom_1 = dataset_info['atom_types'][atom_types_srt[0]]
            atom_2 = dataset_info['atom_types'][atom_types_srt[1]]
            
            distance_diff = get_distance_to_connection(atom_1, atom_2, distance)
            distances[i, j] = distance_diff
            if distance_diff <= 0:
                A[i, j] = 1
                A[j, i] = 1
                
    return A, distances


def get_disconnection_penalty(coords, features, mol_sizes, dataset_name='qm9', use_h=True):
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

        A, distances = get_adjacency(mol_coords, mol_types, dataset_info)
        components = get_graph_components(A)
        if len(components) > 1:
            for component in components:
                dist_to_other = distances[component]
                dist_to_other[:, component] = np.inf
                v, u = np.unravel_index(dist_to_other.argmin(), dist_to_other.shape)
                penalty += dist_to_other[v, u]

        penalties[i] = penalty

    return penalties