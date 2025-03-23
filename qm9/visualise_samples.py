from matplotlib.colors import LinearSegmentedColormap
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
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


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


# Plotting
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0, alpha=alpha, shade=True)

def rgb_to_hex(rgb):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.to(dtype=torch.long).clamp(0, 255)
    else:
        assert isinstance(rgb, np.ndarray)
        rgb = rgb.astype(dtype=np.long).clamp(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info):


    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    colors_dic = np.array([
        [255, 170, 170],
        [120, 120, 120],
        [100, 149, 237],
        [255, 105, 105],
        [144, 238, 144]
    ], dtype=np.float32)
    radius_dic = 0.75 * np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2

    if atom_type.dim() == 1:
        areas = area_dic[atom_type]
        radii = radius_dic[atom_type]
        colors = colors_dic[atom_type]
    else:
        areas = atom_type @ area_dic
        radii = atom_type @ radius_dic
        colors = atom_type @ colors_dic
        atom_type = atom_type.argmax(dim=-1)
    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, rgb_to_hex(c), 0.9)
    else:
        ax.scatter(x, y, z, s=areas, c=[rgb_to_hex(col) for col in colors])
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]], dataset_info['atom_decoder'][s[1]])
            if 'qm9' in dataset_info['name'] or 'qm7b' in dataset_info['name']:
                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
                line_width = (3 - 2) * 2 * 2
            elif dataset_info['name'] == 'geom':
                draw_edge_int = bond_analyze.geom_predictor(pair, dist)
                line_width = 2
            else:
                raise Exception('Wrong dataset_info name')
            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linewidth=line_width * linewidth_factor, c="#AAAAAA", alpha=0.7)
                
    


def plot_data3d(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False, bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False
    ax.set_axis_off()  # Hide axis
    ax.set_facecolor('#FFFFFF00')

    # Create a custom colormap for the gradient
    colors = [(0, 0, 1), (1, 1, 1)]  # Blue to White
    n_bins = 100  # Number of bins for color transition
    cmap_name = 'blue_white_gradient'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Generate a gradient image
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    gradient = np.vstack([gradient])

    
    ax.set_axis_off()  # Remove axis for a clean background


    # if bg == 'black':
    #     ax.xaxis.line.set_color("black")
    #     ax.yaxis.line.set_color("black")
    #     ax.zaxis.line.set_color("black")
    # else:
    #     ax.xaxis.line.set_color("white")
    #     ax.yaxis.line.set_color("white")
    #     ax.zaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name'] or 'qm7b' in dataset_info['name']:
        max_value = positions.abs().max().item()

        axis_lim = 3
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()

        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50


    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        return fig
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
