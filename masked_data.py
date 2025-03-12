from math import log
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from torch.utils import data as td
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from os import path
from multiprocessing import cpu_count

from args import args
from utils.diffusion import demean_using_mask
from utils.files import check_hash_file, hash_file
from utils.qm9 import charge_to_idx, ensure_qm9_raw_data, ensure_qm9_raw_excluded, ensure_qm9_raw_splits, ensure_qm9_raw_thermo, ensure_qm9_processed

from qm9_meta import QM9_WITH_H, QM9_WITHOUT_H

QM9Attributes = Literal["index", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", "omega1", "zpve_thermo", "U0_thermo", "U_thermo", "H_thermo", "G_thermo", "Cv_thermo"]
QM9ProcessedData = dict[str | Literal["num_atoms", "classes", "charges", "positions", "one_hot", QM9Attributes], torch.Tensor]
class QM9ProcessedDataClass:
    def __init__(self, data: QM9ProcessedData):
        self._data = data
    def __getitem__(self, key: str):
        return self._data[key]
    def __setitem__(self, key: str, val):
        self._data[key] = val
    def to_(self, *args, **kwargs):
        for key, val in self._data.items():
            if isinstance(val, torch.Tensor):
                self._data[key] = val.to(*args, **kwargs)

class QM9MaskedDataset(td.Dataset):
    def __init__(self, use_h: bool, split: Literal["train", "valid", "test"]):
        self.num_atom_types=QM9_WITH_H["num_atom_types"] if use_h else QM9_WITHOUT_H["num_atom_types"]
        self.max_nodes=QM9_WITH_H["largest_molecule_size"] if use_h else QM9_WITHOUT_H["largest_molecule_size"]
        self.size_histogram=QM9_WITH_H["molecule_size_histogram"] if use_h else QM9_WITHOUT_H["molecule_size_histogram"]
        
        assert split in ["train", "valid", "test"]
        
        self.split = split
        self.use_h = use_h
        
        self.atom_types = torch.tensor([1,6,7,8,9], dtype=torch.long)
        self.num_atom_types = torch.tensor(5, dtype=torch.long)
        
        if not use_h:
            raise NotImplementedError  # XXX: implement no hydrogens
        
        self._processed_path = path.join(args.original_data_dir, "qm9", f"{split}.npz")        
        self._data: QM9ProcessedData = {key: torch.from_numpy(val) for key, val in np.load(self._processed_path).items()}
        self._data["one_hot"] = self._data["charges"][:,:,None] == self.atom_types
        self._len  = len(self._data["num_atoms"])

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index) -> QM9ProcessedData:
        processed_data: QM9ProcessedData = {key: val[index] for key, val in self._data.items()}
        return processed_data

def _collate_fn(data: list[QM9ProcessedData]):
    keys = data[0].keys()
    stacked_data = {}
    
    # stack tensors so that the first dimension is batch
    for key in keys:
        list_of_values = [molecule[key] for molecule in data]
        if data[0][key].dim() == 0:
            stacked_data[key] = torch.stack(list_of_values)
        else:
            stacked_data[key] = torch.nn.utils.rnn.pad_sequence(list_of_values, batch_first=True, padding_value=0)

    # remove unnecessary columns of zeros
    dims_to_keep = stacked_data["charges"].sum(dim=0) > 0
    for key in keys:
        if stacked_data[key].dim() > 1:
            stacked_data[key] = stacked_data[key][:, dims_to_keep]
            
    # indicate where atoms are located
    # [B, N] where B is batch size and N is size of largest molecule; node_mask[i, j] == True if and only if the jth atom of the ith molecule is present
    node_mask = stacked_data["charges"] > 0 
    stacked_data["node_mask"] = node_mask[:, :, None]
    batch_size, max_n_atoms = node_mask.shape
    
    # indicate where edges are located
    # [B, N, N] where edge_mask[i, j, k] == True if and only if there is an edge between the j and kth positions of molecule i. no edge between an atom and itself. no edge for atoms that are not present. 
    edge_mask = node_mask[:, None, :] * node_mask[:, :, None]
    edge_mask *= ~torch.eye(max_n_atoms, dtype=torch.bool)[None]
    stacked_data["edge_mask"] = edge_mask.flatten(start_dim=1)  # flatten and add extra dim
    
    # add another dimension to the charges
    stacked_data["charges"] = stacked_data["charges"][:, :, None]
    
    # finally, demean the atom positions
    stacked_data["positions"] = demean_using_mask(stacked_data["positions"], stacked_data["node_mask"])
    stacked_data["batch_size"] = stacked_data["positions"].shape[0]
    
    return QM9ProcessedDataClass(stacked_data)


def get_masked_qm9_dataloader(use_h: bool, split: Literal["train", "valid", "test"], batch_size: int, prefetch_factor: int|None=None, num_workers=0, pin_memory=True, shuffle:bool|None=None):
    if shuffle is None:
        shuffle = split == "train"
    return td.DataLoader(dataset=QM9MaskedDataset(use_h=use_h, split=split), batch_size=batch_size, collate_fn=_collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=num_workers, shuffle=shuffle)