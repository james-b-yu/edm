# TODO: Re-write code for evaluation - calculating the proper upper bound on NLL and metrics
# TODO: Rich's suggestion carrying out evaluations and experiments on pre-trained model checkpoints provided by the authors 
# and comparing the results to your own models

import torch
import torch.nn.functional as F
from models.base import EGCL, EGCLConfig
from models.base import EGNN, EGNNConfig
import pickle
from data import get_qm9_dataloader
from argparse import Namespace
import re

# checkpoint used is inside the gdrive uploaded by James
# insert checkpoint wish to evaluate here
argument_path = '2025-03-10 Partially Pretrained Own Architecture/args_epoch_576.pkl'
checkpoint_path = '2025-03-10 Partially Pretrained Own Architecture/model_epoch_576.pth'

def load_model(checkpoint_path, args_path):
    # Load the args
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    if isinstance(args, Namespace):
        args = vars(args)
        
    config = EGNNConfig(
        num_layers=args.get("num_layers"),
        hidden_d=256, 
        features_d = 6, # TODO: hard coded as this was not in the checkpoint but can add in dynamically once we have a more stable model.py
        node_attr_d=args.get("node_attr_d", 0),
        edge_attr_d=args.get("edge_attr_d", 0),
        use_tanh=args.get("use_tanh", True),
        tanh_range=args.get("tanh_range", 15.0),
        use_resid=args.get("use_resid", False),
        dataset_name=args.get("dataset", "qm9").split('_')[0],
        use_h=("_no_h" not in args.get("dataset", "qm9"))
    )
    
    model = EGNN(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    # strip any unnecessary prefixes from the checkpoint keys
    new_checkpoint = {}
    for k, v in checkpoint.items():
        new_key = re.sub(r"^egnn\.", "", k)  # Remove "egnn." prefix if present as current model does not expect
        new_checkpoint[new_key] = v

    # Load the adjusted checkpoint
    model.load_state_dict(new_checkpoint, strict=False)  # Allow some mismatches
    model.eval()
    return model

def load_test_data():
    dataloader = get_qm9_dataloader(use_h=True, split="test", batch_size=64)
    batch = next(iter(dataloader))  # Get a single batch
    batch.to_(torch.device("cpu"))  # Ensure it's on the correct device

    # modify `batch.one_hot` in-place
    if batch.one_hot.shape[1] == 5:  # If missing 1 feature, add a zero-padding column this was causing issues as model.py in this branch was not the 
        # same size as in checkpoint - think a time dim was added? Can remove once sure that dim will recieve 
        batch.one_hot = torch.cat([batch.one_hot, torch.zeros((batch.one_hot.shape[0], 1), device=batch.one_hot.device)], dim=-1)

    return batch

def compute_nll(model, test_data):
    coords = test_data.coords  
    features = test_data.one_hot
    edges = test_data.edges
    reduce = test_data.reduce
    demean = test_data.demean
    batch_size = test_data.n_nodes.shape[0]  # no of molecules

    # make sure time is [N, 1] so that we can concatinate
    time = torch.zeros((features.shape[0], 1), dtype=torch.float32, device=coords.device)

    with torch.no_grad():
        output_coords, output_features = model(
            test_data.n_nodes,  
            coords,
            features,
            edges,
            reduce,
            demean,
            time,
        )

        # get gaussian log-likelihood for the coords
        D = coords.shape[1]  # Number of coordinate dimensions - should be in 3D
        sigma = 1.0  # sd used in paper
        squared_diff = ((output_coords - coords) ** 2).sum(dim=-1)

        log_prob_coords = - (squared_diff / (2 * sigma**2)) - (D / 2) * torch.log(torch.tensor(2 * torch.pi * sigma**2, device=coords.device))

        assert torch.all((log_prob_coords <= 0) ), "Warning: Probability for coords is not between 0 and 1 as log probability is positive"

        nll_coords = log_prob_coords.sum() / batch_size

        # cross-Entropy loss for the features 
        probs_features = torch.softmax(output_features, dim=-1)  # Converting to probabilities
        assert torch.all((probs_features >= 0) & (probs_features <= 1)), " Probability for features is not between 0 and 1"

        labels = features.argmax(dim=-1)  # Convert one-hot to class indices
        nll_features = -F.cross_entropy(output_features, labels, reduction="sum") / batch_size

        # total NLL
        total_nll = nll_coords + nll_features

    print(f"NLL Coords: {nll_coords.item()}, NLL Features: {nll_features.item()}, Total NLL: {total_nll.item()}")
    return total_nll.item()

if __name__ == "__main__":
    model = load_model(checkpoint_path, argument_path)
    test_data = load_test_data()
    nll = compute_nll(model, test_data)
    print(f"Negative Log-Likelihood (NLL): {nll}")
    
# save
 # number_molecules_stable = 0
        # number_atoms_stable = 0
        # number_atoms_stable_total = 0
        # number_atoms = 0
        
        # if args.dataset == 'qm9':
        #     remove_h = False
        # elif args.dataset == 'qm9_no_h':
        #     remove_h = True
        
        # dataset_info = get_dataset_info(remove_h)
        
        # for s in range(len(samples)):
            
        #     #get valid and unique metrics
        #     m = BasicMolecularMetrics(dataset_info)
        #     (validity, uniqueness, _) , _= m.evaluate([(torch.from_numpy(s[0]), torch.from_numpy(s[1]).argmax(dim=-1)) for s in samples])
            
        #     # print(f"molecule {s+1} has {len(samples[s][0])} atoms")
        #     number_atoms += len(samples[s][0])
        #     # xyz coords of each atom
            
        #     # print(f"xyz coords: {samples[0][0]}")
        #     coords = torch.tensor(samples[s][0], device=args.device)
        
        #     # one hot encoding of atom type H, C, O, N, F
        #     # print(f"one hot encoding of atoms: {samples[0][1]}")
        #     one_hot = torch.tensor(samples[s][1], device=args.device)
        
        #     # predicted valencies of each atom
        #     # print(f"charges: {samples[0][2]}")
        #     charges = torch.tensor(samples[s][2], device=args.device)

        #     node_mask = torch.ones(len(samples[s][0]), dtype=torch.bool, device=args.device)  # Shape: [num_atoms]
            
        #     # Ensure inputs are in the correct format
        #     one_hot = one_hot.unsqueeze(0)  # Add batch dimension
        #     charges = charges.unsqueeze(0).unsqueeze(-1)  # Add batch and last dimension
        #     coords = coords.unsqueeze(0)  # Add batch dimension
        #     node_mask = node_mask.unsqueeze(0)  # Add batch dimension
            
        #     samples_torch = [(torch.from_numpy(s[0]), torch.from_numpy(s[1]).argmax(dim=-1)) for s in samples]
        #     res = [check_stability(s[0], s[1], dataset_info) for s in samples_torch] # tqdm(
        #     molecule_stabililty = np.mean([r[0] for r in res])
        #     atom_stability = np.sum([r[1] for r in res]) / np.sum([r[2] for r in res])

        #     # print(f"Molecule stability was {molecule_stabililty:.2f} and atom stability was {atom_stability:.2f}")
            
        #     # atom_stability_wrong = compute_atom_stability(one_hot, charges, coords, node_mask, dataset_info)
        #     # number_atoms_stable_per_molecule = np.sum(atom_stability_wrong)
            
        #     # number_atoms_stable_total += number_atoms_stable_per_molecule
        
        # # stability from correct author code
        # percentage_atoms_stable = atom_stability * 100
        # percentage_molecules_stable = molecule_stabililty * 100
