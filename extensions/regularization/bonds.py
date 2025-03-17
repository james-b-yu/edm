import numpy as np
from configs.bond_config import BONDS_1, BONDS_2, BONDS_3, MARGIN_1, MARGIN_2, MARGIN_3

def get_bond_order(atom_1, atom_2, distance, check_exists=False):
    distance = 100 * distance

    if check_exists:
        if atom_1 not in BONDS_1:
            return 0, np.inf
        if atom_2 not in BONDS_1[atom_1]:
            return 0, np.inf

    threshold_1 = BONDS_1[atom_1][atom_2] + MARGIN_1
    if distance < threshold_1:
        # Check if atoms in bonds2 dictionary.
        if atom_1 in BONDS_2 and atom_2 in BONDS_2[atom_1]:
            threshold_2 = BONDS_2[atom_1][atom_2] + MARGIN_2
            if distance < threshold_2:
                if atom_1 in BONDS_3 and atom_2 in BONDS_3[atom_1]:
                    threshold_3 = BONDS_3[atom_1][atom_2] + MARGIN_3
                    if distance < threshold_3:
                        return 3, 0                     # Triple
                return 2, threshold_3 - distance        # Double
        return 1, threshold_2 - distance                # Single
    return 0, threshold_1 - distance                    # No bond


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order
    

def get_adjacency(coords, atom_type, dataset_info):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    A = np.zeros((len(x), len(x)))
    distances = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))

            atom_1 = dataset_info['atom_decoder'][atom_type[i]]
            atom_2 = dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])
            
            if dataset_info['name'] == 'qm9':
                bond_order, dist_diff = get_bond_order(atom_1, atom_2, dist)
            elif dataset_info['name'] == 'geom':
                bond_order, dist_diff = geom_predictor(pair, dist)
            else:
                raise Exception(f'Wrong dataset {dataset_info['name']}')
            
            distances[i, j] = dist_diff
            if bond_order > 0:
                A[i, j] = 1
                
    return A, distances

