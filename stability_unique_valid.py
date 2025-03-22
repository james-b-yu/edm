
import torch
from configs.datasets_config import get_dataset_info
from qm9.rdkit_functions import BasicMolecularMetrics
import numpy as np
from qm9 import bond_analyze
import logging
logging.getLogger("rdkit").setLevel(logging.ERROR)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppresses all RDKit warnings
import datetime

# too slow
# def check_stability(positions, atom_type, dataset_info, debug=False):

#     assert len(positions.shape) == 2
#     assert positions.shape[1] == 3
#     atom_decoder = dataset_info['atom_decoder']
#     x = positions[:, 0]
#     y = positions[:, 1]
#     z = positions[:, 2]

#     nr_bonds = np.zeros(len(x), dtype='int')

# ##########bottleneck################
#     for i in range(len(x)):
#         for j in range(i + 1, len(x)):
#             p1 = np.array([x[i].cpu().numpy(), y[i].cpu().numpy(), z[i].cpu().numpy()])
#             p2 = np.array([x[j].cpu().numpy(), y[j].cpu().numpy(), z[j].cpu().numpy()])

#             dist = np.sqrt(np.sum((p1 - p2) ** 2))

#             atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
#             pair = sorted([atom_type[i], atom_type[j]])
#             if dataset_info['name'] == 'qm7b' or dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
#                 order = bond_analyze.get_bond_order(atom1, atom2, dist)
#             elif dataset_info['name'] == 'geom':
#                 order = bond_analyze.geom_predictor(
#                     (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
#             nr_bonds[i] += order
#             nr_bonds[j] += order
#     nr_stable_bonds = 0
#     for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
#         possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        
#         if type(possible_bonds) == int:
#             is_stable = possible_bonds == nr_bonds_i
#         else:
#             is_stable = nr_bonds_i in possible_bonds
#         if not is_stable and debug:
#             print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
#         nr_stable_bonds += int(is_stable)

#     molecule_stable = nr_stable_bonds == len(x)
#     return molecule_stable, nr_stable_bonds, len(x)

def check_stability(positions, atom_type, dataset_info, debug=False):
    """
    Fast version: Computes bond distances using vectorized operations.
    Same input/output as original function.
    """
    assert len(positions.shape) == 2 and positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    atom_type = atom_type.cpu()
    positions = positions.cpu()

    coords = positions.numpy()
    atom_type_np = atom_type.numpy()

    n = len(coords)
    nr_bonds = np.zeros(n, dtype=int)

    # -------- Vectorized pairwise distances --------
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # Shape (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)  # Shape (n, n)

    # -------- Loop only over upper triangle --------
    for i in range(n):
        for j in range(i + 1, n):
            dist = dists[i, j]
            atom1 = atom_decoder[atom_type_np[i]]
            atom2 = atom_decoder[atom_type_np[j]]
            pair = sorted([atom_type_np[i], atom_type_np[j]])

            if dataset_info['name'] in {'qm7b', 'qm9', 'qm9_first_half', 'qm9_second_half'}:
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            else:
                order = 0

            nr_bonds[i] += order
            nr_bonds[j] += order

    # -------- Stability check --------
    nr_stable_bonds = 0
    for i in range(n):
        atom_name = atom_decoder[atom_type_np[i]]
        possible_bonds = bond_analyze.allowed_bonds[atom_name]
        is_stable = (
            nr_bonds[i] == possible_bonds
            if isinstance(possible_bonds, int)
            else nr_bonds[i] in possible_bonds
        )
        if debug and not is_stable:
            print(f"Invalid bonds for {atom_name} with {nr_bonds[i]} bonds")
        nr_stable_bonds += int(is_stable)

    molecule_stable = (nr_stable_bonds == n)
    return molecule_stable, nr_stable_bonds, n


def compute_stability_unique_and_valid(samples, args):

    if args.dataset == 'qm9':
        remove_h = False
    elif args.dataset == 'qm9_no_h':
        remove_h = True

    dataset_info = get_dataset_info(remove_h)
    logfile = f"sampling_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(logfile, "a+") as f:
        for s in range(len(samples)):
            
            #get valid and unique metrics
            m = BasicMolecularMetrics(dataset_info)
            (validity, uniqueness, _) , _= m.evaluate([(torch.from_numpy(s[0]), torch.from_numpy(s[1]).argmax(dim=-1)) for s in samples])

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            samples_torch = [(torch.from_numpy(s[0]).to(device), torch.from_numpy(s[1]).argmax(dim=-1).to(device)) for s in samples]

            res = [check_stability(s[0], s[1], dataset_info) for s in samples_torch] # tqdm(
            molecule_stabililty = np.mean([r[0] for r in res])
            atom_stability = np.sum([r[1] for r in res]) / np.sum([r[2] for r in res])
            
            
            valid_to_log = validity * 100
            valid_unique_to_log = uniqueness * validity * 100
            # logging

            # f.write(f"molecule: {s}\n")
            # f.write(f"molecule stab: {molecule_stabililty:.4f} %\n")
            # f.write(f"atom stab: {atom_stability:.4f} %\n")
            # f.write(f"valid: {valid_to_log:.4f} %\n")
            # f.write(f"unique and valid: {valid_unique_to_log:.4f} %\n")
            # f.write(f"\n")
            

        total_molecules = len(samples)
        
        # stability from correct author code
        percentage_atoms_stable = atom_stability * 100
        percentage_molecules_stable = molecule_stabililty * 100
        percentage_of_valid_are_unique = uniqueness * 100
        percentage_valid = validity * 100
        valid_and_unique_percentage = (percentage_of_valid_are_unique /100 ) * (validity) 
        
        f.write(f"############# Final Results #############\n")
        f.write(f"percentage_atoms_stable: {percentage_atoms_stable:.4f} %\n")
        f.write(f"percentage_molecules_stable: {percentage_molecules_stable:.4f} %\n")
        f.write(f"percentage_of_valid_are_unique: {percentage_of_valid_are_unique:.4f} %\n")
        f.write(f"percentage_valid : {percentage_valid :.4f} %\n")
        f.write(f"valid_and_unique_percentage : {valid_and_unique_percentage :.4f} %\n")
    
    return percentage_atoms_stable, percentage_molecules_stable, percentage_valid, valid_and_unique_percentage, res



# updated for faster
# import torch
# from configs.datasets_config import get_dataset_info
# from qm9.rdkit_functions import BasicMolecularMetrics
# import numpy as np
# from qm9 import bond_analyze
# import datetime
# import multiprocessing as mp  # Import multiprocessing explicitly



# def check_stability(positions, atom_type, dataset_info, debug=False):
#     """
#     Checks the stability of a molecule.
#     """
#     positions = positions.cpu()  # 🛑 Ensure positions are on CPU
#     atom_type = atom_type.cpu()  # 🛑 Ensure atom_type is on CPU

#     assert len(positions.shape) == 2
#     assert positions.shape[1] == 3
#     atom_decoder = dataset_info['atom_decoder']

#     # Ensure tensors are on CPU before conversion
#     x = positions[:, 0].cpu().numpy()
#     y = positions[:, 1].cpu().numpy()
#     z = positions[:, 2].cpu().numpy()
#     print(x)
#     print(y)
#     print(z)

#     nr_bonds = np.zeros(len(x), dtype='int')

#     # 🛑 The bottleneck: Optimize using vectorized operations later
#     for i in range(len(x)):
#         for j in range(i + 1, len(x)):
#             p1 = np.array([x[i], y[i], z[i]])
#             p2 = np.array([x[j], y[j], z[j]])
#             dist = np.linalg.norm(p1 - p2)

#             atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
#             order = bond_analyze.get_bond_order(atom1, atom2, dist)
#             print(f"Checking bond order: {atom1}-{atom2} at distance {dist} -> order: {order}")

#             nr_bonds[i] += order
#             nr_bonds[j] += order

#     nr_stable_bonds = sum(nr_bonds[i] == bond_analyze.allowed_bonds[atom_decoder[atom_type[i]]] for i in range(len(x)))

#     molecule_stable = nr_stable_bonds == len(x)
#     return molecule_stable, nr_stable_bonds, len(x)

# def compute_stability_unique_and_valid(samples, args):
#     dataset_info = get_dataset_info(args.dataset == 'qm9_no_h')

#     logfile = f"sampling_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
#     with open(logfile, "a") as f:
#         m = BasicMolecularMetrics(dataset_info)
#         validity_tuple = m.evaluate([(torch.from_numpy(s[0]), torch.from_numpy(s[1]).argmax(dim=-1)) for s in samples])

#         validity, uniqueness = validity_tuple[0][0], validity_tuple[0][1]  # Extract only the first two values

#         # Move tensors to CPU before multiprocessing
#         samples_torch = [(torch.from_numpy(s[0]).cpu(), torch.from_numpy(s[1]).argmax(dim=-1).cpu()) for s in samples]

#         res = []  # 🔥 Initialize `res` to avoid UnboundLocalError

#         # Run multiprocessing only in the main process
#         if __name__ == "__main__":
#             with mp.Pool(processes=mp.cpu_count()) as pool:
#                 res = pool.starmap(check_stability, [(s[0], s[1], dataset_info) for s in samples_torch])
#                 print(res)

#         # If `res` is empty, set default values (avoid errors in np.mean)
#         if len(res) == 0:
#             res = [(False, 0, 1)]  # Default: 0 stability if no samples processed

#         molecule_stability = np.mean([r[0] for r in res])
#         atom_stability = np.sum([r[1] for r in res]) / np.sum([r[2] for r in res])
        
#         percentage_atoms_stable = atom_stability * 100
#         percentage_molecules_stable = molecule_stability * 100
#         percentage_of_valid_are_unique = uniqueness * 100
#         percentage_valid = validity * 100
#         valid_and_unique_percentage = (percentage_of_valid_are_unique /100 ) * (validity) 

#         f.write(f"############# Final Results #############\n")
#         f.write(f"percentage_atoms_stable: {percentage_atoms_stable:.4f} %\n")
#         f.write(f"percentage_molecules_stable: {percentage_molecules_stable:.4f} %\n")
#         f.write(f"percentage_of_valid_are_unique: {percentage_of_valid_are_unique:.4f} %\n")
#         f.write(f"percentage_valid : {percentage_valid :.4f} %\n")
#         f.write(f"valid_and_unique_percentage : {valid_and_unique_percentage :.4f} %\n")

#     percentage_atoms_stable = atom_stability * 100
#     percentage_molecules_stable = molecule_stability * 100
#     percentage_valid = validity * 100
#     valid_and_unique_percentage = (uniqueness * validity) * 100

#     return percentage_atoms_stable, percentage_molecules_stable, percentage_valid, valid_and_unique_percentage, res


