# to run use:
# ./run.py --extension vanilla --pipeline test --checkpoint ./checkpoints/pretrained --no-wandb --device=cpu

from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path
import torch.nn.functional as F
import numpy as np
# from diffusion import polynomial_schedule # sort import later

from masked_model import MaskedEDM, get_config_from_args

@torch.no_grad()


# OG
def compute_atom_stability(one_hot, charges):
    """
    Computes atomic stability based on valency rules.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [N, num_atom_classes].
        charges (torch.Tensor): Atomic charges [N, 1].

    Returns:
        torch.Tensor: Boolean tensor indicating whether each atom is stable.
    """
    VALENCY = torch.tensor([1, 4, 3, 2, 1], dtype=torch.float32, device=one_hot.device)
    predicted_valencies = (one_hot * VALENCY).sum(dim=-1) + charges.squeeze()
    is_stable = (predicted_valencies >= 0) & (predicted_valencies <= VALENCY.max())
    
    return is_stable

def compute_molecule_stability(one_hot, charges, n_nodes):
    """
    Computes molecule stability.

    A molecule is stable if all its atoms are stable.

    Args:
        one_hot (torch.Tensor): Atom types as one-hot encoding [N, num_atom_classes].
        charges (torch.Tensor): Atomic charges [N, 1].
        n_nodes (torch.Tensor): Number of atoms per molecule [B].

    Returns:
        float: Percentage of stable molecules in the batch.
    """
    n_nodes = n_nodes.sum(dim=1).long()

    is_stable_atoms = compute_atom_stability(one_hot, charges)
    
    # Split into molecules
    stable_molecules = []
    index = 0
    for n in n_nodes:
        molecule_stability = is_stable_atoms[index : index + n].all()  # True if all atoms are stable
        stable_molecules.append(molecule_stability)
        index += n
    
    stable_molecules = torch.tensor(stable_molecules, dtype=torch.float32, device=one_hot.device)
    
    return stable_molecules.mean().item() * 100  # Convert to percentage

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
    total_atoms = 0
    total_stable_molecules = 0
    total_molecules = 0
    total_samples = 0
    num_batches = 0
    
    config = get_config_from_args(args, dl.dataset.num_atom_types)  # type:ignore
    
    model = MaskedEDM(config)
    model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=config.device))
    model.eval()
    
     # Create the noise schedule
    noise_schedule = polynomial_schedule_just_sigma(args.num_steps, config.device)

    
    # so we have got config and model
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
        sigma = 0.5
        squared_diff = ((pred_eps_coord - eps_coord) ** 2).sum(dim=-1)
        # log_prob_coords = - (squared_diff / (2 * sigma**2)) - (D / 2) * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=config.device))
        log_prob_coords = - (squared_diff / (2 * sigma**2)) - (D / 2) * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=config.device) + 1e-8) # can include a small epsilon for log stability

        nll_coords = log_prob_coords.sum() / batch_size
        
        # Cross-Entropy loss for atomic features
        # Reshape prediction tensor to [batch_size * num_elements, num_classes]
        pred_eps_feat = pred_eps_feat.view(-1, 6)  # Collapse first two dims, keep num_classes
        labels = eps_feat.argmax(dim=-1) 
        # Flatten labels to match the first dimension of pred_eps_feat
        labels = labels.view(-1)  # Flatten to [batch_size * num_elements]
        # print(labels)
        # Compute cross-entropy loss
        nll_features = F.cross_entropy(pred_eps_feat, labels, reduction="sum") / batch_size

        
        total_nll_per_batch += nll_coords + nll_features
        # total_nll_acc += (nll_coords + nll_features) * batch_size
        total_nll_acc += (nll_coords.detach() + nll_features.detach()) * batch_size # save on memory using detach()

        
        batch_stability = compute_atom_stability(data['one_hot'], data['charges']) # , data['node_mask']
        total_stable_atoms += batch_stability.float().mean().item() * batch_size
        total_atoms += batch_stability.numel()  # Correctly count total atoms
        
        # Compute molecule stability
        n_nodes_per_molecule = data["node_mask"].sum(dim=1).long()  # Sum mask values per molecule to get atom count
        batch_molecule_stability = compute_molecule_stability(data['one_hot'], data['charges'], data["node_mask"])
        # Get atomic stability for all atoms
        is_stable_atoms = compute_atom_stability(data['one_hot'], data['charges']) # , data['node_mask'

        # Convert atomic stability into molecule stability
        stable_molecules = []
        index = 0
        for n in n_nodes_per_molecule:
            molecule_stability = is_stable_atoms[index : index + n].all().item()  # Check if all atoms are stable
            stable_molecules.append(molecule_stability)
            index += n

        # Convert list to tensor for easy sum
        stable_molecules = torch.tensor(stable_molecules, dtype=torch.float32, device=args.device)

        # Sum up number of stable molecules
        total_stable_molecules += stable_molecules.sum().item()
        total_molecules += n_nodes_per_molecule.shape[0]

        num_batches += 1
        total_samples += batch_size
        avg_batch_nll = total_nll_per_batch
        
        # DEBUG
        # Inside the loop...
        # print(f"Batch {idx}:")
        # print(f"  n_nodes: {n_nodes_per_molecule}")
        # print(f"  eps_coord shape: {eps_coord.shape}")
        # print(f"  pred_eps_coord shape: {pred_eps_coord.shape}")
        # print(f"  sigma: {sigma}")
        # print(f"  squared_diff: {squared_diff.mean()}")
        # print(f"  log_prob_coords: {log_prob_coords.mean()}")
        # print(f"  pred_eps_feat shape: {pred_eps_feat.shape}")
        # print(f"  labels shape: {labels.shape}")
        # print(f"  is_stable_atoms: {is_stable_atoms}")

        
        # print progress bar with value for each batch
        pbar.set_description(f"Batch MSE {mse:.2f}. Running Batch Average NLL: {avg_batch_nll}")

    # compute_molecule_stability(data['one_hot'], data['charges'], data['n_nodes'])
        
    overall_atom_stability = (total_stable_atoms / total_samples) * 100 if total_samples > 0 else 0
    overall_molecule_stability = (total_stable_molecules / total_molecules) * 100 if total_molecules > 0 else 0

    
    avg_nll = total_nll_acc / total_samples
    print(f"Total samples: {total_samples}")
    print(f"Final results: MSE: {mse:.2f} \n NLL: {avg_nll:.2f}")
    print(f"Overall Atom Stability: {overall_atom_stability:.2f}%")
    print(f"Overall Molecule Stability: {overall_molecule_stability:.2f}%")
    return(mse, avg_nll)