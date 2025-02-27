from os import path, makedirs
import os
from pathlib import Path
from typing import Iterable, Sequence
from tqdm import tqdm
import numpy as np
import json


from .misc import is_int
from .files import check_hash_directory, check_hash_file, delete_folder, hash_directory, hash_file, urlretrieve, tar_extractall
from args import args


def get_crg_cls_dicts(use_h: bool):
    """get dicts converting between charge and class indices

    Args:
        use_h (bool): whether we are using hydrogens

    Returns:
        tuple[dict[int, int], dict[int, int]]: charge to class, class to charge
    """
    if use_h:
        crg_to_cls = {
            1: 0,  # H
            6: 1,  # C
            7: 2,  # N
            8: 3,  # O
            9: 4   # F
        }
    else:
        crg_to_cls = {
            6: 0,  # C
            7: 1,  # N
            8: 2,  # O
            9: 3   # F
        }
        
    cls_to_crg = {val: key for key, val in crg_to_cls.items()}
    return crg_to_cls, cls_to_crg

def charge_to_idx(charges: np.ndarray, use_h: bool):
    """given a np array of charges, return an np array of the same shape but with charges replaced with class indices

    Args:
        charges (np.ndarray):
    """
    orig_dtype = charges.dtype

    crg_to_cls, _ = get_crg_cls_dicts(use_h)
    where_cls: list[tuple[np.ndarray, int]] = []
    
    for crg, cls in crg_to_cls.items():
        where_cls.append((charges == crg, cls))
    
    for where, cls in where_cls:
        charges[where] = cls
    
    assert charges.dtype == orig_dtype
    return charges
        

def process_xyz_qm9(xyz_path: str, use_h: bool, therm_dict: dict[str, dict[str, float]]):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    max_num_nodes = 29 if use_h else 9
    
    with open(xyz_path, "r", encoding="utf8") as f:
        xyz_lines = [line for line in f.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    # now calculate thermo energy
    charge_counts = np.unique(atom_charges, return_counts=True)
    for target, therm_target in therm_dict.items():
        therm_res = 0.0
        for z, z_count in zip(*charge_counts):
            assert z != 0
            assert str(z) in therm_target
            therm_res += therm_target[str(z)] * z_count
        
        mol_props[f"{target}_thermo"] = therm_res

    # now get rid of hydrogens if we are using the non-hydrogen dataset
    if not use_h:
        num_atoms = num_atoms - sum([1 if c == 1 else 0 for c in atom_charges])
        atom_positions = [p for i, p in enumerate(atom_positions) if atom_charges[i] != 1]
        atom_charges = [c for c in atom_charges if c != 1]

    charges = np.array(atom_charges + [0] * (max_num_nodes - len(atom_charges)), dtype=np.int64)
    classes = charge_to_idx(charges, use_h)

    molecule = {'num_atoms': np.array(num_atoms, dtype=np.int64), 'charges': charges, 'classes': classes, 'positions': np.array(atom_positions + [[0.0, 0.0, 0.0]] * (max_num_nodes - len(atom_charges)), dtype=np.float32)}
    molecule.update({key: np.array(value, dtype=np.float32) for key, value in mol_props.items()})

    return molecule

def _download_and_extract_data(data_path: str):
    """downloads qm9.xyz data and extracts to data_path

    Args:
        data_path (str): path to which we want to extract
    """
    tar_path = data_path + ".tar.bz2"
    
    if not (Path(tar_path).is_file() and check_hash_file(tar_path, args.qm9_raw_xyz_tar_md5)):
        urlretrieve(url=args.qm9_data_url, filename=tar_path, desc="Downloading raw QM9 data")
        
    tar_extractall(tar_path=tar_path, extract_path=data_path, desc="Extracting raw QM9 data")
    
    
    

def ensure_qm9_raw_data(parent_path: str):
    """Checks to see whether qm9 raw data files are extracted to disk at `{parent_path}/raw/xyz`. If not, retrieves and extracts them

    Args:
        parent_path (str): path to the containing folder
    """
    
    data_path = path.join(parent_path, "xyz")
    do_download = False

    if not Path(data_path).is_dir():
        do_download = True
    elif not check_hash_directory(data_path, args.qm9_raw_xyz_dir_md5, desc="Checking integrity of raw QM9 data"):
        delete_folder(data_path, desc="Deleting raw QM9 data")
        do_download = True
    
    if do_download:
        _download_and_extract_data(data_path)
        
def ensure_qm9_raw_excluded(parent_path: str):
    """Checks to see whether qm9 raw `excluded.txt` file is located at `{parent_path}/raw/excluded.txt`. If not, retrieves it

    Args:
        parent_path (str): path to the containing folder
    """
    
    excluded_path = path.join(parent_path, "excluded.txt")
    
    if not (Path(excluded_path).is_file() and check_hash_file(excluded_path, args.qm9_raw_excluded_txt_md5)):
        urlretrieve(url=args.qm9_excluded_url, filename=excluded_path, desc="Downloading raw QM9 excluded.txt")
        
def ensure_qm9_raw_splits(parent_path: str):
    """Checks to see whether qm9 raw `splits.npz` file is located at `{parent_path}/raw/splits.npz`. If not, generates it

    Args:
        parent_path (str): path to the containing folder
    """
    splits_path = path.join(parent_path, "splits.npz")
    
    if Path(splits_path).is_file() and check_hash_file(splits_path, args.qm9_raw_spilts_npz_md5):
        return
    
    excluded_path = path.join(parent_path, "excluded.txt")
    
    assert Path(excluded_path).is_file() and check_hash_file(excluded_path, args.qm9_raw_excluded_txt_md5)
    gdb9_txt_excluded = excluded_path  # make alias so that it works with the copied code below

    """The following code is copy-pasted directly from `https://github.com/risilab/cormorant/blob/master/src/cormorant/data/prepare/qm9.py` lines 99--128. It makes sure we use the correct splits
    """
    """BEGIN COPIED CODE"""
    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]
    
    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))
    
    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]
    """END COPIED CODE"""
    
    np.savez(splits_path, train=train, valid=valid, test=test)
    
    
def ensure_qm9_raw_thermo(parent_path: str):
    """Checks to see whether qm9 raw `thermo.json` file is located at `{parent_path}/raw/thermo.json`. If not, generates it

    Args:
        parent_path (str): path to the containing folder
    """
    thermo_path = path.join(parent_path, "thermo.json")
    
    if Path(thermo_path).is_file() and check_hash_file(thermo_path, args.qm9_raw_thermo_json_md5):
        return
    
    atomref_path = path.join(parent_path, "atomref.txt")
    if not (Path(atomref_path).is_file() and check_hash_file(atomref_path, args.qm9_raw_atomref_txt_md5)):
        urlretrieve(url=args.qm9_atomref_url, filename=atomref_path, desc="Downloading atomref data")
        
    gdb9_txt_thermo = atomref_path  # alias so that we can work with the copied code below
    
    """The following code is copy-pasted directly from `https://github.com/risilab/cormorant/blob/master/src/cormorant/data/prepare/qm9.py` lines 158--172. It retrieves thermochemical energies for every molecule
    """
    """BEGIN COPIED CODE"""
    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in tqdm(list(f), desc="Retrieving thermochemical energies", leave=False):
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)
    """END COPIED CODE"""
    
    with open(thermo_path, "w", encoding="utf8") as f:
        json.dump(therm_energy, f)
        
def _collate(data: np.ndarray):
    """Given a list of dicts with `np.ndarray`s as values, return a dict of `np.ndarray`s collating them together. This assumes every dict has the same keys.

    Args:
        data (np.ndarray[dict[str, np.ndarray]]):

    Returns:
        _type_: _description_
    """
    assert len(data) > 0
    return {
        key: np.stack([d[key] for d in data]) for key in tqdm(data[0], leave=False, desc="Stacking arrays")
    }

def ensure_qm9_processed(disk_path: str, use_h: bool):    
    makedirs(path.join(disk_path), exist_ok=True)
    raw_path = path.join(disk_path, "raw")
    makedirs(path.join(raw_path), exist_ok=True)
    
    ensure_qm9_raw_data(raw_path)
    ensure_qm9_raw_excluded(raw_path)
    ensure_qm9_raw_splits(raw_path)
    ensure_qm9_raw_thermo(raw_path)
    
    xyz_dir_path = path.join(raw_path, "xyz")
    splits_path = path.join(raw_path, "splits.npz")
    therm_path = path.join(raw_path, "thermo.json")
    
    with open(therm_path, "r", encoding="utf8") as f:
        therm_dict = json.load(f)
    
    xyz_file_paths = []
    
    for root, _, filenames in os.walk(xyz_dir_path):
        for filename in tqdm(filenames, unit="file", leave=False, desc="Gathering raw QM9 XYZ files"):
            xyz_file_paths.append(path.join(root, filename))
            
    xyz_files_processed = np.array([process_xyz_qm9(p, use_h, therm_dict) for p in tqdm(xyz_file_paths, unit="file", leave=False, desc="Processing raw QM9 XYZ files")])
    splits = np.load(splits_path)
    
    processed = {key: _collate(xyz_files_processed[val]) for key, val in tqdm(splits.items(), unit="dataset", leave=False, desc="Collating datasets")}
    
    for key, val in processed.items():
        np.savez(path.join(disk_path, f"{key}_{"h" if use_h else "no_h"}.npz"), **val)  # type: ignore