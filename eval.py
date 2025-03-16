import torch
import torch.nn.functional as F
from model import EGNN, EGNNConfig
import pickle
from data import get_qm9_dataloader
from argparse import Namespace
import re

# Checkpoint paths

# argument_path = ''
checkpoint_path = 'checkpoints/pretrained/model.pth'

# def load_model(checkpoint_path, args_path):
#     with open(args_path, "rb") as f:
#         args = pickle.load(f)

#     if isinstance(args, Namespace):
#         args = vars(args)
        
#     # config = EGNNConfig(
#     #     num_layers=args.get("num_layers"),
#     #     hidden_dim=256,
#     #     num_atom_types=5,
#     #     # node_attr_d=args.get("node_attr_d", 0),
#     #     # edge_attr_d=args.get("edge_attr_d", 0),
#     #     # use_tanh=args.get("use_tanh", True),
#     #     # tanh_range=args.get("tanh_range", 15.0)
#     # )
    
#     model = EGNN(config)
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     model.load_state_dict(checkpoint, strict=False)
#     model.eval()
#     return model

def get_test_dataloader():
    return get_qm9_dataloader(use_h=True, split="test", batch_size=64)

def compute_nll(model, batch):
    coords = batch.coords  
    features = batch.one_hot  
    edges = batch.edges
    reduce = batch.reduce
    demean = batch.demean
    batch_size = batch.n_nodes.shape[0]
    time = torch.zeros((features.shape[0], 1), dtype=torch.float32, device=coords.device)
    
    with torch.no_grad():
    # output_coords, output_features = model(batch.n_nodes, coords, features, edges, reduce, demean, time)
        output_coords, output_features = model(coords, features, time, edges, reduce, demean)
                                        #  def forward(self, coord, one_hot, charge, time_int, node_mask, edge_mask):
        # Gaussian log-likelihood for coordinates
        D = coords.shape[1]  
        sigma = 1.0
        squared_diff = ((output_coords - coords) ** 2).sum(dim=-1)
        log_prob_coords = - (squared_diff / (2 * sigma**2)) - (D / 2) * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=coords.device))
        nll_coords = log_prob_coords.sum() / batch_size
        
        # Cross-Entropy loss for atomic features
        labels = features.argmax(dim=-1)
        nll_features = -F.cross_entropy(output_features, labels, reduction="sum") / batch_size
        
        total_nll = nll_coords + nll_features
    
    return total_nll.item()

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

def evaluate_model(model, dataloader):
    """
    Evaluates the model by computing NLL, atom stability, and molecule stability.
    """
    total_nll = 0.0
    total_stable_atoms = 0
    total_atoms = 0
    total_stable_molecules = 0
    total_molecules = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch.to_(torch.device("cpu"))
            
            # Compute NLL
            batch_nll = compute_nll(model, batch)
            total_nll += batch_nll
            
            # Compute atom stability
            batch_stability = compute_atom_stability(batch.one_hot, batch.charges)
            total_stable_atoms += batch_stability.sum().item()
            total_atoms += batch.one_hot.shape[0]
            
            # Compute molecule stability
            batch_molecule_stability = compute_molecule_stability(batch.one_hot, batch.charges, batch.n_nodes)
            total_stable_molecules += batch_molecule_stability * batch.n_nodes.shape[0]  # Convert % to count
            total_molecules += batch.n_nodes.shape[0]
            
            num_batches += 1
    
    avg_nll = total_nll / num_batches
    overall_atom_stability = (total_stable_atoms / total_atoms) * 100 if total_atoms > 0 else 0
    overall_molecule_stability = (total_stable_molecules / total_molecules) if total_molecules > 0 else 0

    print(f"Average Negative Log-Likelihood (NLL): {avg_nll:.4f}")
    print(f"Overall Atom Stability: {overall_atom_stability:.2f}%")
    print(f"Overall Molecule Stability: {overall_molecule_stability:.2f}%")

# if __name__ == "__main__":
#     model = load_model(checkpoint_path, argument_path)
#     test_dataloader = get_test_dataloader()
#     evaluate_model(model, test_dataloader)