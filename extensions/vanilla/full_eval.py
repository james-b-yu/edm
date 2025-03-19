from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from qm9 import bond_analyze
from configs.datasets_config import get_dataset_info
import pickle
import os

@torch.no_grad()

def remove_mean_with_mask(coords, node_mask):
    """Centers molecular coordinates by subtracting the mean."""
    valid_mask = node_mask > 0  # Boolean mask for valid atoms

    masked_coords = coords * valid_mask  # Zero out invalid atoms

    # Compute the mean ONLY over valid atoms
    num_valid = valid_mask.sum(dim=1, keepdim=True)  # Count valid atoms per molecule
    mean = masked_coords.sum(dim=1, keepdim=True) / (num_valid + 1e-8)  # Avoid division by zero

    # Subtract the mean only from valid atoms
    centered_coords = coords - mean * valid_mask  # Ensure padding remains 0

    return centered_coords

def remove_mean(coords):
    """Centers molecular coordinates by subtracting the mean."""
    mean = coords.mean(dim=1, keepdim=True)  # Compute mean along the correct dimension
    centered_coords = coords - mean  # Subtract mean from all atoms
    return centered_coords

def normalize_coords(coords):
    """Centers molecular coordinates and ensures unit-scale."""
    mean = coords.mean(dim=1, keepdim=True)  # Compute mean
    std = coords.std(dim=1, keepdim=True) + 1e-8  # Compute std and prevent divide-by-zero
    normalized_coords = (coords - mean) / std  # Apply both centering and scaling
    return normalized_coords



def compute_atom_stability(one_hot, charges, coords, node_mask, dataset_info):
    """
    Computes atomic stability based on inferred bonds using RDKit + empirical distance-based bond rules.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [batch, num_atoms, num_atom_classes].
        charges (torch.Tensor): Atomic charges [batch, num_atoms, 1].
        coords (torch.Tensor): Atomic 3D coordinates [batch, num_atoms, 3].
        node_mask (torch.Tensor): Mask indicating valid atoms [batch, num_atoms].
        dataset_info (dict): Dataset configuration info, input True/False for removing H (ie True = remove H, False = with H)

    Returns:
        torch.Tensor: Boolean tensor [batch, num_atoms] indicating whether each atom is stable.
    """
    print(f"coords before normalization: {coords}")
    coords = normalize_coords(coords)  # Ensure coordinates are normalized
    print(f"coords after normalization: {coords}")

    batch_size, num_atoms, num_atom_classes = one_hot.shape
    atom_decoder = dataset_info['atom_decoder']

    atom_types_batch = torch.argmax(one_hot, dim=-1).cpu().numpy()
    charges_batch = charges.cpu().numpy().squeeze(-1)
    coords_batch = coords.cpu().numpy()
    node_mask_batch = node_mask.cpu().numpy()

    stability_results = []

    for b in range(batch_size):
        atom_types = atom_types_batch[b]
        charges = charges_batch[b]
        coords = coords_batch[b]

        valid_mask = node_mask_batch[b].astype(bool)  # Convert mask to boolean
        positions = coords[valid_mask]  
        atom_types_b = atom_types[valid_mask]  

        valid_atoms = atom_types_b != 0  
        positions = positions[valid_atoms]
        atom_types_b = atom_types_b[valid_atoms]
        nr_bonds = np.zeros(len(positions), dtype=int)  # Reset per molecule
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        print(f"positions: {positions}")
        print(f"atom_types_b: {atom_types_b}")

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1 = np.array([x[i], y[i], z[i]])
                p2 = np.array([x[j], y[j], z[j]])
                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                atom1, atom2 = atom_decoder[atom_types_b[i]], atom_decoder[atom_types_b[j]]

                order = bond_analyze.get_bond_order(atom1, atom2, dist)

                nr_bonds[i] += order
                nr_bonds[j] += order

                print(f"Updated bonds: Atom {i} ({atom1}) = {nr_bonds[i]}, Atom {j} ({atom2}) = {nr_bonds[j]}")

        stable_atoms = []
        for i in range(len(atom_types_b)):
            possible_bonds = bond_analyze.allowed_bonds.get(atom_decoder[atom_types_b[i]], [])

            predicted_bonds = nr_bonds[i]
            print(f"predicting: {predicted_bonds} and expecting: {possible_bonds}")

            if isinstance(possible_bonds, list):
                is_stable = predicted_bonds in possible_bonds
            else:
                is_stable = predicted_bonds == possible_bonds

            stable_atoms.append(is_stable)

        stability_results.append(torch.tensor(stable_atoms, dtype=torch.bool, device=one_hot.device))

    return torch.nn.utils.rnn.pad_sequence(stability_results, batch_first=True, padding_value=False) 

def compute_molecule_stability(one_hot, charges, coords, node_mask, dataset_info):
    """
    Computes molecule stability based on atomic stability.

    A molecule is stable if *all* its atoms are stable.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [batch, num_atoms, num_atom_classes].
        charges (torch.Tensor): Atomic charges [batch, num_atoms, 1].
        coords (torch.Tensor): Atomic 3D coordinates [batch, num_atoms, 3].
        node_mask (torch.Tensor): Mask indicating valid atoms [batch, num_atoms].

    Returns:
        tuple: (Number of stable molecules, total number of molecules)
    """
    batch_size = one_hot.shape[0]  # Number of molecules in batch

    # Compute atomic stability
    is_stable_atoms = compute_atom_stability(one_hot, charges, coords, node_mask, dataset_info)

    # Get the number of nodes per molecule from the node_mask.
    n_nodes_per_molecule = node_mask.sum(dim=1).long()

    # Split into molecules using n_nodes_per_molecule.
    stable_molecules = 0
    index = 0
    for n in n_nodes_per_molecule:
        molecule_stability = is_stable_atoms[index: index + n]
        is_molecule_stable = molecule_stability.numel() > 0 and molecule_stability.all().item()
        if is_molecule_stable:
            stable_molecules += 1
        index += n

    return stable_molecules, batch_size


# need to add this into NLL as currently sigma just = 1.0
def polynomial_schedule_just_sigma(timesteps: int, device: torch.device|str, power=2., s=1e-5):
    """
    ADAPTED FROM https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L39
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    def clip_noise_schedule(alphas2, clip_value=0.001):
        alphas2 = np.clip(alphas2, a_min=clip_value, a_max=1.)
        alphas2 = alphas2 / alphas2[0]
        return alphas2


def run_eval(args: Namespace, dl: DataLoader):
    print("This calculates the Atom stability and Molecule Stability against the test data.")
    nlls = []
    total_stable_atoms = 0
    total_stable_molecules = 0
    total_molecules = 0
    total_atoms = 0
    total_samples = 0
    
    config = get_config_from_args(args, dl.dataset.num_atom_types)  # type:ignore
    
    model = MaskedEDM(config)
    model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=config.device))
    model.eval()
    
    # Create the noise schedule
    noise_schedule = polynomial_schedule_just_sigma(args.num_steps, config.device)
    
    for idx, data in enumerate(pbar := tqdm(dl)):
        total_nll_per_batch = 0 # reset per batch
        nll_coords = 0 # ensure is reset each batch
        nll_features = 0 # ensure is reset each batch
        ######################
        #calculate the MSE
        ######################
        
        data._to(dtype=torch.float32, device=args.device)
        batch_size = data["batch_size"]  # note this may vary over batch because the final batch of a dataset may be smaller
        time_int = torch.randint(low=0, high=args.num_steps + 1, size=(batch_size, ), device=args.device, dtype=torch.long)
        
        (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = model.get_eps_and_predicted_eps(data["positions"], data["one_hot"], data["charges"], time_int, data["node_mask"], data["edge_mask"])
        
        sq_coord_err = (eps_coord - pred_eps_coord) ** 2
        sq_feat_err  = (eps_feat - pred_eps_feat) ** 2
        
        mse = torch.concat([sq_coord_err, sq_feat_err], dim=-1).mean()
        
        ######################
        # calculate the NLL
        ######################
        
        D = eps_coord.shape[1]  
        sigma_t = polynomial_schedule_just_sigma(args.num_steps, config.device)
        # when using proper noise schedule NLL explodes
        sigma = 0.5 # noise_schedule
        squared_diff = ((pred_eps_coord - eps_coord) ** 2).sum(dim=-1)
        log_prob_coords = - (squared_diff / (2 * sigma**2)) - (D / 2) * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=config.device) + 1e-8) # can include a small epsilon for log stability

        nll_coords = log_prob_coords.sum() / batch_size
        
        # Cross-Entropy loss for atomic features
        # Reshape prediction tensor to [batch_size * num_elements, num_classes]
        pred_eps_feat = pred_eps_feat.view(-1, 6)  # Collapse first two dims, keep num_classes
        labels = eps_feat.argmax(dim=-1) 
        # Flatten labels to match the first dimension of pred_eps_feat
        labels = labels.view(-1)  # Flatten to [batch_size * num_elements]

        # Compute cross-entropy loss
        nll_features = F.cross_entropy(pred_eps_feat, labels, reduction="sum") / batch_size

        total_nll_per_batch += nll_coords.detach() + nll_features.detach()
        nlls.append(total_nll_per_batch)
        
        ######################
        # calculate atom stability
        ######################
        
        batch_stability = compute_atom_stability(data['one_hot'], data['charges'], pred_eps_coord, data['node_mask'], get_dataset_info())

        # Count number of True values across all tensors
        batch_num_stable_atoms = sum(tensor.sum().item() for tensor in batch_stability)
        total_stable_atoms += batch_num_stable_atoms
        total_atoms += batch_stability.numel()  # Correctly count total atoms

        ######################
        # calculate molecule stability
        ######################
        
        stable_molecules, total_molecules = compute_molecule_stability(data['one_hot'], data['charges'], pred_eps_coord, data['node_mask'])

        # # Sum up number of stable molecules
        total_stable_molecules += stable_molecules
        total_molecules += total_molecules
        
        running_atom_stab = (total_stable_atoms / total_atoms) * 100
        running_molecule_stab = (total_stable_molecules / total_molecules) * 100
        total_samples += batch_size
        
        # print progress bar with value for each batch
        pbar.set_description(f"Batch MSE {mse:.2f}. Running NLL: {total_nll_per_batch}. Running atom stability: {running_atom_stab} %. Running molecule stab: {running_molecule_stab} %")
        
    overall_atom_stability = (total_stable_atoms / total_atoms) * 100 if total_samples > 0 else 0
    overall_molecule_stability = (total_stable_molecules / total_samples) * 100 if total_molecules > 0 else 0

    avg_nll = sum(nlls) / len(nlls)
    print(f"Total samples: {total_molecules}")
    print(f"Final results: MSE: {mse:.2f} \n NLL: {avg_nll:.2f}")
    print(f"Overall Atom Stability: {overall_atom_stability:.2f}%")
    print(f"Overall Molecule Stability: {overall_molecule_stability:.2f}%")

    return(mse, avg_nll)