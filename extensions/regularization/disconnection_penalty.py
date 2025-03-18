import torch
import numpy as np
from collections import deque
from extensions.regularization import bonds
from configs.dataset_config import get_dataset_info

def bfs(start, A):
    queue = deque()
    queue.append(start)
    visited = set()
    reachable = [start]

    while (len(queue) > 0):
        v = queue.popleft()
        reachable.append(v)
        for (u, edge) in enumerate(A[v]):
            if edge and u not in visited:
                visited.add(u)
                queue.append(u)

    return reachable


def find_graph_components(A):
    v = 0
    visited = set()
    component = 0
    components = {}

    while len(components) < A.shape[0]:
        if v not in visited:
            reachable = bfs(v, A)
            visited.update(reachable)
            components[component] = reachable
            component += 1
        v += 1

    return components


def get_disconnection_penalty(batch_coords, batch_features, dataset_name='qm9', use_h=True):
    batch_penalties = torch.zeros((batch_coords.shape[0],))
    dataset_info = get_dataset_info(dataset_name, use_h)
    num_types = len(dataset_info['atom_types'])

    batch_coords = batch_coords.view(-1, 3)
    batch_features = batch_features[:, :num_types].view(-1, num_types).type(torch.float32)
    atom_types = torch.argmax(batch_coords, dim=1).numpy()

    for (i, coords) in enumerate(batch_coords):
        penalty = 0.
        A, distances = bonds.get_adjacency(coords, atom_types[i], dataset_info)
        components = find_graph_components(A)
        if len(components) > 1:
            for (_, nodes) in components.items():
                dist_to_other = distances[nodes]
                dist_to_other[:, nodes] = np.inf
                v, u = np.unravel_index(dist_to_other.argmin(), dist_to_other.shape)
                penalty += distances[v, u]
        batch_penalties[i] = penalty

    return batch_penalties