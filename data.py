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

@dataclass
class EDMDatasetItem:
    """Attributes:
        n_nodes (int): n, number of atoms in the molecule
        coords (torch.Tensor): [n, 3] giving the 3D coordinates (x) of each atom in the molecule
        one_hot (torch.Tensor): [n, num_atom_types] giving the one_hot encoded classes of each atom in the molecule
        charges (torch.Tensor): [n, 1] giving the number of protons in each atom in the molecule
        size_log_prob (float): prior log probability of having a molecule of this size
    """
    n_nodes: int
    coords: torch.Tensor
    one_hot: torch.Tensor
    charges: torch.Tensor
    size_log_prob: float

@dataclass
class EDMBaseDataloaderItem:
    """A class representing a batch of molecules. This base class contains types of information common to both the masked and flattened representations, although the actual shapes of these tensors will differ between the two
    """
    num_atoms: torch.Tensor
    coords: torch.Tensor
    one_hot: torch.Tensor
    charges: torch.Tensor
    size_log_probs: torch.Tensor
    
    def to_(self, *args, **kwargs):
        """calls `torch.Tensor.to` on all tensors, assigning the results
        """
        self.coords = self.coords.to(*args, **kwargs)
        self.one_hot = self.one_hot.to(*args, **kwargs)
        self.charges = self.charges.to(*args, **kwargs)
        self.size_log_probs = self.size_log_probs.to(*args, **kwargs)
        
        # force long
        self.num_atoms = self.num_atoms.to(dtype=torch.long, device=self.coords.device)

    @property
    def batch_size(self):
        return self.num_atoms.numel()

    def __getitem__(self, key):
        if key == "batch_size":
            return self.batch_size
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Key '{key}' not found.")

@dataclass
class EDMDataloaderItem(EDMBaseDataloaderItem):
    """
     Note that we use "flattened" notation, in that coords and features are one long tensor of size [N, 3] or [N, num_atom_types] respectively, where N is the number of atoms across the entire batch. This allows each molecule to have a different number of atoms and we keep track of this dynamically.

    We treat each batch as one single large "graph": each individual molecule in the batch has fully connected nodes. Therefore in order to represent the edges in the batch we only require tensors of outer length NN, where NN = (n_nodes ** 2).sum() is the number of edges in the batch.

    Attributes:
        n_nodes (torch.Tensor): [B] where B is the batch size and each bth entry is the number of atoms in the bth molecule. Let N = n_nodes.sum() be the number of atoms in the entire batch and NN = (n_nodes ** 2).sum(), the number of edges in the batch.
        coords (torch.Tensor): [N, 3], where N = n_nodes.sum(), giving the 3D coordinates of each atom in the batch
        one_hot (torch.Tensor): [N, num_atom_types] giving the class of the each atom in the batch
        charges (torch.Tensor): [N, 1] giving the charge of each atom in the batch
        edges (torch.Tensor): (NN, 2) tensor, where NN = (n_nodes ** 2).sum(). Each row of this tensor gives the indices of atoms within the batch for which there is a directed edge. Note that we do allow edges from a node to itself as this simplifies dealing with the ordering of the edges across the code
        reduce (torch.Tensor): (N, NN) binary float matrix where reduce[i, j] = 1.0 if and only if there is an edge from atom i->j
        batch_mean (torch.Tensor): (B, N) float matrix such that given an [N, dim] matrix A, the operation mean @ A gives an [B, dim] matrix where information is averaged over molecules
        batch_sum (torch.Tensor): (B, N) float matrix such that given an [N, dim] matrix A, the operation mean @ A gives an [B, dim] matrix where information is summed over molecules
        demean (torch.Tensor): (NN, NN) float matrix such that given an [NN, dim] matrix A, the operatation demean @ A gives an [NN, dim] matrix where molecule-wise centre of masses are zero
        expand_idx (torch.Tensor): (N) if a is a vector of length B with elements corresponding to each molecule in the batch, then a[expand_idx] is a vector of length N where each element is repeated for every atom in the molecule
        size_log_probs (torch.Tensor): [B] prior log probs of molecule size
    """
    
    edges: torch.Tensor
    reduce: torch.Tensor
    batch_mean: torch.Tensor
    batch_sum: torch.Tensor
    demean: torch.Tensor
    expand_idx: torch.Tensor
    
    def to_(self, *args, **kwargs):
        """calls `torch.Tensor.to` on all tensors, assigning the results
        """
        super().to_(*args, **kwargs)
    
        self.reduce = self.reduce.to(*args, **kwargs)
        self.batch_mean = self.batch_mean.to(*args, **kwargs)
        self.batch_sum = self.batch_sum.to(*args, **kwargs)
        self.demean = self.demean.to(*args, **kwargs)
        
        # force long type on these
        self.edges = self.edges.to(dtype=torch.long, device=self.coords.device)
        self.expand_idx = self.expand_idx.to(dtype=torch.long, device=self.coords.device)
        
@dataclass
class EDMMaskedDataloaderItem(EDMBaseDataloaderItem):
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    
    def to_(self, *args, **kwargs):
        """calls `torch.Tensor.to` on all tensors, assigning the results
        """
        super().to_(*args, **kwargs)
    
        self.node_mask = self.node_mask.to(*args, **kwargs)
        self.edge_mask = self.edge_mask.to(*args, **kwargs)


QM9Attributes = Literal["index", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", "omega1", "zpve_thermo", "U0_thermo", "U_thermo", "H_thermo", "G_thermo", "Cv_thermo"]
QM9ProcessedData = dict[Literal["num_atoms", "classes", "charges", "positions", "one_hot", QM9Attributes], torch.Tensor]

class EDMDataset(ABC, td.Dataset):
    def __init__(self, num_atom_types, max_nodes, atom_types, size_histogram):
        """initialise EDM dataset

        Args:
            num_atom_types (int): number of data features
            max_nodes (int): maximum number of atoms per molecule
            atom_types (list[int]): list of atom charges that appear in the dataset (sort this ascending)
            size_histogram (dict[int, int]|None): count of molecule sizes
        """
        super().__init__()
        assert num_atom_types > 2

        self.num_atom_types = num_atom_types
        self.max_nodes = max_nodes
        self.size_histogram = size_histogram
        self.atom_types = torch.tensor(atom_types, dtype=torch.long)
        
        hist_norm = sum(size_histogram.values())
        self.log_size_histogram_probs = {key: log((val / hist_norm) + 1e-30) for key, val in size_histogram.items()}

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> EDMDatasetItem:
        pass

class QM9Dataset(EDMDataset):
    def __init__(self, use_h: bool, split: Literal["train", "valid", "test"]):
        super().__init__(
            num_atom_types=QM9_WITH_H["num_atom_types"] if use_h else QM9_WITHOUT_H["num_atom_types"],
            max_nodes=QM9_WITH_H["largest_molecule_size"] if use_h else QM9_WITHOUT_H["largest_molecule_size"],
            atom_types=QM9_WITH_H["atom_types"] if use_h else QM9_WITHOUT_H["atom_types"],
            size_histogram=QM9_WITH_H["molecule_size_histogram"] if use_h else QM9_WITHOUT_H["molecule_size_histogram"]
        )
        
        assert split in ["train", "valid", "test"]
        
        self.split = split
        self.use_h = use_h
        
        processed_filename = f"{split}_{"h" if use_h else "no_h"}"
        self._processed_path = path.join(args.data_dir, "qm9", f"{processed_filename}.npz")
        if not (Path(self._processed_path).is_file() and check_hash_file(self._processed_path, getattr(args, f"qm9_{processed_filename}_npz_md5"))):
            ensure_qm9_processed(path.join(args.data_dir, "qm9"), use_h)
        
        self._data: QM9ProcessedData = {key: torch.from_numpy(val) for key, val in np.load(self._processed_path).items()}
        self._data["one_hot"] = self._data["charges"][:,:,None] == self.atom_types
        self._len  = len(self._data["num_atoms"])

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index) -> EDMDatasetItem:
        n_nodes = self._data["num_atoms"][index]
        coords = self._data["positions"][index, :n_nodes]
        charges = self._data["charges"][index, :n_nodes]
        one_hot = self._data["one_hot"][index, :n_nodes]
        size_log_prob = self.log_size_histogram_probs[int(n_nodes)]
        
        return EDMDatasetItem(
            n_nodes = int(n_nodes),
            coords = coords,
            charges = charges,
            one_hot = one_hot,
            size_log_prob = size_log_prob
        )

def _collate_fn(data: list[EDMDatasetItem]):
        n_nodes = torch.tensor([d.n_nodes for d in data], dtype=torch.int64)
        coords = torch.cat([d.coords for d in data])
        one_hot = torch.cat([d.one_hot for d in data])
        charges = torch.cat([d.charges for d in data])[:, None]
        size_log_probs = torch.tensor([d.size_log_prob for d in data], dtype=torch.float32)

        _n_nodes_cumsum = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), n_nodes]), dim=0)

        edges = torch.cat([torch.cartesian_prod(torch.arange(b_n.item(), dtype=torch.int64), torch.arange(b_n.item(), dtype=torch.int64)) + _n_nodes_cumsum[b] for b, b_n in enumerate(n_nodes)], dim=0)
        reduce = torch.block_diag(*[torch.block_diag(*[torch.ones((int(n), )).scatter_(0, torch.tensor([m]), 0) for m in range(n)]) for n in n_nodes])
        demean = torch.block_diag(*[torch.eye(int(n), dtype=torch.float32) - (torch.ones((int(n), int(n)), dtype=torch.float32) / n) for n in n_nodes])
        expand_idx = torch.cat([torch.ones(size=(int(n), ), dtype=torch.long) * idx for idx, n in enumerate(n_nodes)])
        batch_mean = torch.block_diag(*[(1/float(n)) * torch.ones(size=(1, int(n)), dtype=torch.float32) for n in n_nodes])
        batch_sum = torch.block_diag(*[torch.ones(size=(1, int(n)), dtype=torch.float32) for n in n_nodes])
        coords = demean @ coords  # immediately demean the coords

        return EDMDataloaderItem(num_atoms=n_nodes, coords=coords, one_hot=one_hot, charges=charges, edges=edges, reduce=reduce, batch_mean=batch_mean, batch_sum=batch_sum, demean=demean, expand_idx=expand_idx, size_log_probs=size_log_probs)

def _masked_collate_fn(data: list[EDMDatasetItem]):
    n_nodes = torch.tensor([d.n_nodes for d in data], dtype=torch.int64)
    size_log_probs = torch.tensor([d.size_log_prob for d in data], dtype=torch.float32)
    max_n_nodes = int(n_nodes.max())
    coords  = torch.nn.utils.rnn.pad_sequence([d.coords for d in data], batch_first=True, padding_value=0)
    one_hot = torch.nn.utils.rnn.pad_sequence([d.one_hot for d in data], batch_first=True, padding_value=0)
    charges = torch.nn.utils.rnn.pad_sequence([d.charges for d in data], batch_first=True, padding_value=0)
    
    node_mask = charges > 0
    edge_mask = node_mask[:, None, :] * node_mask[:, :, None]
    edge_mask *= ~torch.eye(max_n_nodes, dtype=torch.bool)[None]
    edge_mask = edge_mask.flatten(start_dim=1)
    
    # add extra dimension to things
    node_mask = node_mask[:, :, None]
    charges = charges[:, :, None]
    
    # set convert to floating point tensors
    edge_mask = edge_mask.to(dtype=torch.float32)
    node_mask = node_mask.to(dtype=torch.float32)
    
    # finally demean coordinates
    coords = demean_using_mask(coords, node_mask)
    batch_size = coords.shape[0]
    
    return EDMMaskedDataloaderItem(
        num_atoms=n_nodes,
        coords=coords,
        one_hot=one_hot,
        charges=charges,
        node_mask=node_mask,
        edge_mask=edge_mask,
        size_log_probs=size_log_probs
    )

def get_qm9_dataloader(use_h: bool, split: Literal["train", "valid", "test"], batch_size: int, prefetch_factor: int|None=None, num_workers = 0, pin_memory=True, shuffle:bool|None=True):
    if shuffle is None:
        shuffle = split == "train"
    return td.DataLoader(dataset=QM9Dataset(use_h=use_h, split=split), batch_size=batch_size, collate_fn=_collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=num_workers, shuffle=shuffle)

def get_masked_qm9_dataloader(use_h: bool, split: Literal["train", "valid", "test"], batch_size: int, prefetch_factor: int|None=None, num_workers=0, pin_memory=True, shuffle:bool|None=None):
    if shuffle is None:
        shuffle = split == "train"
    return td.DataLoader(dataset=QM9Dataset(use_h=use_h, split=split), batch_size=batch_size, collate_fn=_masked_collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=num_workers, shuffle=shuffle)