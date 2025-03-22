import torch
import numpy as np
import os
import glob
import random
from datetime import datetime
import matplotlib
import imageio

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from configs.datasets_config import qm9_with_h, geom_with_h
from data import get_qm9_dataloader
from qm9 import bond_analyze




# Files
def save_xyz_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None):
    os.makedirs(path, exist_ok=True)

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        filename = os.path.join(path, f"{name}_{batch_i + id_from:03d}.txt")
        with open(filename, "w") as f:
            f.write("%d\n\n" % atomsxmol[batch_i])
            atoms = torch.argmax(one_hot[batch_i], dim=1)
            n_atoms = int(atomsxmol[batch_i])
            for atom_i in range(n_atoms):
                atom = atoms[atom_i]
                atom = dataset_info['atom_decoder'][atom]
                f.write("%s %.9f %.9f %.9f\n" % (
                    atom,
                    positions[batch_i, atom_i, 0],
                    positions[batch_i, atom_i, 1],
                    positions[batch_i, atom_i, 2]
                ))



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
    # Ensure path is a directory
    if not os.path.isdir(path):
        raise ValueError(f"Expected a directory for path, got file or invalid path: {path}")

    # Build proper search pattern using os.path.join
    pattern = os.path.join(path, "*.txt")
    files = glob.glob(pattern)

    if shuffle:
        random.shuffle(files)
    return files



# Plotting
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0, alpha=alpha)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.linalg.norm(p1 - p2)
            atom1 = dataset_info['atom_decoder'][atom_type[i]]
            atom2 = dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]], dataset_info['atom_decoder'][s[1]])

            if 'qm9' in dataset_info['name'] or 'qm7b' in dataset_info['name']:
                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                draw_edge_int = bond_analyze.geom_predictor(pair, dist)
            else:
                raise Exception('Wrong dataset_info name')

            if draw_edge_int > 0:
                draw_parallel_bonds(
                    ax, p1, p2,
                    bond_order=draw_edge_int,
                    offset=0.12,
                    color=hex_bg_color,
                    alpha=alpha,
                    linewidth=2
                )


def draw_parallel_bonds(ax, p1, p2, bond_order, offset=0.1, color='white', alpha=1.0, linewidth=2):
    """Draw 1 to 3 parallel bonds between p1 and p2"""
    if bond_order == 1:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                linewidth=linewidth, c=color, alpha=alpha)
    else:
        # Get bond direction vector
        direction = np.array(p2) - np.array(p1)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Get arbitrary perpendicular vector
        perp = np.cross(direction, np.array([0, 0, 1]))
        if np.linalg.norm(perp) < 1e-5:
            perp = np.cross(direction, np.array([0, 1, 0]))
        perp = perp / (np.linalg.norm(perp) + 1e-8)

        # Draw multiple bonds with perpendicular offset
        if bond_order == 2:
            shift = perp * offset
            for sign in [-1, 1]:
                p1_shifted = p1 + sign * shift
                p2_shifted = p2 + sign * shift
                ax.plot([p1_shifted[0], p2_shifted[0]],
                        [p1_shifted[1], p2_shifted[1]],
                        [p1_shifted[2], p2_shifted[2]],
                        linewidth=linewidth, c=color, alpha=alpha)
        elif bond_order == 3:
            shift = perp * offset
            # Center line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    linewidth=linewidth, c=color, alpha=alpha)
            # Offset lines
            for sign in [-1, 1]:
                p1_shifted = p1 + sign * shift
                p2_shifted = p2 + sign * shift
                ax.plot([p1_shifted[0], p2_shifted[0]],
                        [p1_shifted[1], p2_shifted[1]],
                        [p1_shifted[2], p2_shifted[2]],
                        linewidth=linewidth, c=color, alpha=alpha)




def plot_data3d(positions, atom_type, dataset_info,
                camera_elev=0, camera_azim=0,
                save_path=None,
                spheres_3d=False,
                bg='black',
                alpha=1.):
    
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    
    ax.set_facecolor(black if bg == 'black' else white)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    line_color = "black" if bg == 'black' else "white"
    ax.xaxis.line.set_color(line_color)
    ax.yaxis.line.set_color(line_color)
    ax.zaxis.line.set_color(line_color)

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info)

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()

    plt.close()



if __name__ == '__main__':
    # matplotlib.use('qt5agg')
    random.seed(datetime.now().timestamp())
    fname_uid = random.randint(10**6, 10**7)
    task_dataset = 'qm9'

    if task_dataset == 'qm9':
        dataset_info = qm9_with_h
        dataloader = get_qm9_dataloader(use_h=True, split='test', batch_size=1)

        for i, data in enumerate(dataloader):
            positions = data['coords'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            plot_data3d(positions_centered, atom_type, dataset_info=dataset_info, spheres_3d=True, save_path=f"img/molecule_{fname_uid}_{i}.jpg")

    elif task_dataset == 'geom':
        files = load_xyz_files('outputs/data')
        matplotlib.use('macosx')
        for file in files:
            x, one_hot, _ = load_molecule_xyz(file, dataset_info=geom_with_h)

            positions = x.view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = one_hot.view(-1, 16).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            mask = (x == 0).sum(1) != 3
            positions_centered = positions_centered[mask]
            atom_type = atom_type[mask]

            plot_data3d(positions_centered, atom_type, dataset_info=geom_with_h, spheres_3d=False)

    else:
        raise ValueError(task_dataset)
