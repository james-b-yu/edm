
from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from masked_model import MaskedEDM, get_config_from_args
from qm9 import bond_analyze  # Import the bond predictor used by authors

@torch.no_grad()


#rdkit version
def compute_atom_stability(one_hot, charges, coords, node_mask):
    """
    Computes atomic stability based on inferred bonds using RDKit + empirical distance-based bond rules.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [batch, num_atoms, num_atom_classes].
        charges (torch.Tensor): Atomic charges [batch, num_atoms, 1].
        coords (torch.Tensor): Atomic 3D coordinates [batch, num_atoms, 3].
        node_mask (torch.Tensor): Mask indicating valid atoms [batch, num_atoms].

    Returns:
        torch.Tensor: Boolean tensor [batch, num_atoms] indicating whether each atom is stable.
    """
    
    batch_size, num_atoms, num_atom_classes = one_hot.shape
    ATOMIC_NUMS = [1, 6, 7, 8, 9]  # H, C, N, O, F
    valency_limits = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}  

    # Convert tensors to numpy for RDKit processing
    atom_types_batch = torch.argmax(one_hot, dim=-1).cpu().numpy()  # Convert one-hot to atomic numbers
    charges_batch = charges.cpu().numpy().squeeze(-1)  
    coords_batch = coords.cpu().numpy()  
    node_mask_batch = node_mask.cpu().numpy()

    stability_results = []

    for b in range(batch_size):  
        atom_types = atom_types_batch[b]  
        charges = charges_batch[b]  
        coords = coords_batch[b]  
        valid_mask = node_mask_batch[b]  # Mask of valid atoms
        
        mol = Chem.RWMol()  
        atom_map = {}  

        # set-up rdkit molecule
        for i, atom_idx in enumerate(atom_types):
            if atom_idx >= len(ATOMIC_NUMS) or valid_mask[i] == 0:  
                continue  # Ignore padding / invalid atoms

            atomic_num = dataset_info['atom_decoder'][atom_idx]  # Ensure correct atomic number
            atom = Chem.Atom(atomic_num)  
            atom.SetFormalCharge(int(charges[i]))  
            idx = mol.AddAtom(atom)  
            atom_map[i] = idx  

        # Add the 3D coords
        conf = Chem.Conformer(len(atom_map))
        for i, (x, y, z) in enumerate(coords[:len(atom_map)]):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf)

        # Infer bonds using distance-based approach
        nr_bonds = np.zeros(num_atoms, dtype=int)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if valid_mask[i] == 0 or valid_mask[j] == 0:
                    continue  # Skip invalid atoms

                dist = np.linalg.norm(coords[i] - coords[j])
                atom1, atom2 = ATOMIC_NUMS[atom_types[i]], ATOMIC_NUMS[atom_types[j]]
                order = bond_analyze.get_bond_order(atom1, atom2, dist)

                nr_bonds[i] += order
                nr_bonds[j] += order

                # **Add bond to RDKit molecule**
                if order > 0:
                    try:
                        mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.values[int(order)])
                    except:
                        print(f"Warning: Could not add bond {atom1}-{atom2} with order {order}")

        try:
            Chem.SanitizeMol(mol)  # Ensure bonds match valency rules
        except:
            print("Warning: RDKit failed to sanitize the molecule. Check bond assignments.")

        # help with ring structures
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)  # Improve aromatic rings
        except:
            print("Warning: Kekulization failed, may affect stability checks.")

        # **Check Stability**
        expected_valencies = np.array([valency_limits.get(ATOMIC_NUMS[i], -1) for i in atom_types])
        is_stable = (nr_bonds == expected_valencies)  # Must match exactly

        # Convert result back to PyTorch tensor
        stability_results.append(torch.tensor(is_stable, dtype=torch.bool, device=one_hot.device))

    return torch.stack(stability_results)  # (batch_size, num_atoms)


# distance away verions
# def compute_atom_stability(one_hot, charges, coords, node_mask):
#     """
#     Computes atomic stability based on inferred bonds using RDKit + empirical distance-based bond rules.

#     Args:
#         one_hot (torch.Tensor): Atom types as one-hot encoding [batch, num_atoms, num_atom_classes].
#         charges (torch.Tensor): Atomic charges [batch, num_atoms, 1].
#         coords (torch.Tensor): Atomic 3D coordinates [batch, num_atoms, 3].
#         node_mask (torch.Tensor): Mask indicating valid atoms [batch, num_atoms].

#     Returns:
#         torch.Tensor: Boolean tensor [batch, num_atoms] indicating whether each atom is stable.
#     """
#     # print(f"node mask: {node_mask}")
#     # print(f"node mask shape: {node_mask.shape}")
    
    
#     batch_size, num_atoms, num_atom_classes = one_hot.shape
#     ATOMIC_NUMS = [1, 6, 7, 8, 9]  # H, C, N, O, F
#     valency_limits = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}  

#     # Convert tensors to numpy for RDKit processing
#     atom_types_batch = torch.argmax(one_hot, dim=-1).cpu().numpy()  # Convert one-hot to atomic numbers
#     charges_batch = charges.cpu().numpy().squeeze(-1)  
#     coords_batch = coords.cpu().numpy()  
#     node_mask_batch = node_mask.cpu().numpy()

#     stability_results = []

#     for b in range(batch_size):  
#         atom_types = atom_types_batch[b]  
#         charges = charges_batch[b]  
#         coords = coords_batch[b]  
#         valid_mask = node_mask_batch[b]  # Mask of valid atoms
        
#         mol = Chem.RWMol()  
#         atom_map = {}  

#         # Add atoms to RDKit molecule
#         for i, atom_idx in enumerate(atom_types):
#             if atom_idx >= len(ATOMIC_NUMS) or valid_mask[i] == 0:  
#                 continue  # Ignore padding / invalid atoms

#             atomic_num = dataset_info['atom_decoder'][atom_idx]

#             atom = Chem.Atom(atomic_num)  
#             atom.SetFormalCharge(int(charges[i]))  
#             idx = mol.AddAtom(atom)  
#             atom_map[i] = idx  

#         # Add 3D coordinates
#         conf = Chem.Conformer(len(atom_map))
#         for i, (x, y, z) in enumerate(coords[:len(atom_map)]):
#             conf.SetAtomPosition(i, (float(x), float(y), float(z)))
#         mol.AddConformer(conf)

#         # **Infer bonds using distance-based approach
#         nr_bonds = np.zeros(num_atoms, dtype=int)

#         for i in range(len(coords)):
#             for j in range(i + 1, len(coords)):
#                 if valid_mask[i] == 0 or valid_mask[j] == 0:
#                     continue  # Skip invalid atoms

#                 dist = np.linalg.norm(coords[i] - coords[j])
#                 atom1, atom2 = ATOMIC_NUMS[atom_types[i]], ATOMIC_NUMS[atom_types[j]]
#                 order = bond_analyze.get_bond_order(atom1, atom2, dist)

#                 nr_bonds[i] += order
#                 nr_bonds[j] += order

#         # **CHECK STRICT VALENCY MATCH**
#         expected_valencies = np.array([valency_limits.get(ATOMIC_NUMS[i], -1) for i in atom_types])
#         is_stable = (nr_bonds == expected_valencies)  # Must match exactly

#         # Convert result back to PyTorch tensor
#         stability_results.append(torch.tensor(is_stable, dtype=torch.bool, device=one_hot.device))

#     return torch.stack(stability_results)  # (batch_size, num_atoms)


def compute_molecule_stability(one_hot, charges, coords, node_mask):
    """
    Computes molecule stability based on atomic stability.

    A molecule is stable if *all* its atoms are stable.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [batch, num_atoms, num_atom_classes].
        charges (torch.Tensor): Atomic charges [batch, num_atoms, 1].
        coords (torch.Tensor): Atomic 3D coordinates [batch, num_atoms, 3].
        node_mask (torch.Tensor): Mask indicating valid atoms [batch, num_atoms].

    Returns:
        float: Percentage of stable molecules in the batch.
    """
    batch_size = one_hot.shape[0]  # Number of molecules in batch

    # Compute atomic stability
    is_stable_atoms = compute_atom_stability(one_hot, charges, coords, node_mask)

    # Split into molecules using batch_sizes
    stable_molecules = []
    index = 0
    for n in batch_size:
        molecule_stability = is_stable_atoms[index : index + n].all().item()  # True if all atoms are stable
        stable_molecules.append(molecule_stability)
        index += n
    
    stable_molecules = torch.tensor(stable_molecules, dtype=torch.float32, device=one_hot.device)
    
    return int(stable_molecules), int(batch_size)

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

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return {"sigma": torch.from_numpy(((1. - alphas2) ** 0.5).astype(np.float32)).to(device=device)}


def run_eval(args: Namespace, dl: DataLoader):
    print("This calculates the NLL, Atom stability and Molecule Stability against the test data.")
    total_nll_acc = 0.0
    total_stable_atoms = 0
    total_stable_molecules = 0
    total_molecules = 0
    total_atoms = 0
    total_samples = 0
    num_batches = 0
    
    config = get_config_from_args(args, dl.dataset.num_atom_types)  # type:ignore
    
    model = MaskedEDM(config)
    model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=config.device))
    model.eval()
    
    # Create the noise schedule
    noise_schedule = polynomial_schedule_just_sigma(args.num_steps, config.device)
    print(noise_schedule)
    
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
        sigma = 1 # noise_schedule
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

        
        total_nll_per_batch += nll_coords + nll_features
        total_nll_acc += (nll_coords.detach() + nll_features.detach()) * batch_size # save on memory using detach()
        
        ######################
        # calculate atom stability
        ######################
        
        batch_stability = compute_atom_stability(data['one_hot'], data['charges'], pred_eps_coord, data['node_mask'])
        total_stable_atoms += batch_stability.float().mean().item() * batch_size
        print(total_stable_atoms)
        total_atoms += batch_stability.numel()  # Correctly count total atoms
        print(total_atoms)

        num_batches += 1
        total_samples += batch_size
        avg_batch_nll = total_nll_per_batch

        ######################
        # calculate molecule stability
        ######################
        
        stable_molecules, total_molecules = compute_molecule_stability(data['one_hot'], data['charges'], pred_eps_coord, data['node_mask'])

        # # Sum up number of stable molecules
        total_stable_molecules += stable_molecules
        total_molecules += total_molecules
        
        running_atom_stab = total_stable_atoms / total_atoms
        running_molecule_stab = total_stable_molecules / total_molecules
        
        # print progress bar with value for each batch
        pbar.set_description(f"Batch MSE {mse:.2f}. Running NLL: {avg_batch_nll}. Running atom stability: {running_atom_stab}. Running molecule stab: {running_molecule_stab}")


        
    overall_atom_stability = (total_stable_atoms / total_samples) * 100 if total_samples > 0 else 0
    overall_molecule_stability = (total_stable_molecules / total_molecules) * 100 if total_molecules > 0 else 0

    avg_nll = total_nll_acc / total_samples
    print(f"Total samples: {total_samples}")
    print(f"Final results: MSE: {mse:.2f} \n NLL: {avg_nll:.2f}")
    print(f"Overall Atom Stability: {overall_atom_stability:.2f}%")
    print(f"Overall Molecule Stability: {overall_molecule_stability:.2f}%")

    return(mse, avg_nll)