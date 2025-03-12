import random
import glob
import torch
import numpy as np
from configs.datasets_config import QM9_WITH_H, GEOM_WITH_H
from extensions.regularization import dataset, bond_analyze

def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges
    

def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files
        
        
def get_adjacency_matrix(positions, atom_type, dataset_info):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    A = torch.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                           dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])
            if 'qm9' in dataset_info['name'] or 'qm7b' in dataset_info['name']:
                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                draw_edge_int = bond_analyze.geom_predictor(pair, dist)
            else:
                raise Exception('Wrong dataset_info name')
            draw_edge = draw_edge_int > 0
            if draw_edge:
                A[i, j] = 1
                
    return A


def get_adj_for_dataset(task_dataset='qm9'):
    adj = []

    if task_dataset == 'qm9':
        dataset_info = QM9_WITH_H

        class Args:
            batch_size = 1
            num_workers = 0
            filter_n_atoms = None
            datadir = 'qm9/temp'
            dataset = 'qm9'
            remove_h = False
            include_charges = True

        cfg = Args()

        dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

        for i, data in enumerate(dataloaders['train']):
            positions = data['positions'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()
            A = get_adjacency_matrix(positions, atom_type, dataset_info)
            adj.append(A)

    elif task_dataset == 'geom':
        dataset_info = GEOM_WITH_H
        files = load_xyz_files('outputs/data')

        for file in files:
            x, one_hot, _ = load_molecule_xyz(file, dataset_info)

            positions = x.view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = one_hot.view(-1, 16).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            mask = (x == 0).sum(1) != 3
            positions_centered = positions_centered[mask]
            atom_type = atom_type[mask]
            A = get_adjacency_matrix(positions, atom_type, dataset_info)
            adj.append(A)

    else:
        raise ValueError(dataset)
    
    return adj
