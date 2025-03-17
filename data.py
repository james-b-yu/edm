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
from utils.files import check_hash_file, hash_file
from utils.qm9 import charge_to_idx, ensure_qm9_raw_data, ensure_qm9_raw_excluded, ensure_qm9_raw_splits, ensure_qm9_raw_thermo, ensure_qm9_processed

from configs.dataset_config import QM9, QM9_NO_H

@dataclass
class EDMDatasetItem:
    """Attributes:
        n_nodes (int): n, number of atoms in the molecule
        coords (torch.Tensor): [n, 3] giving the 3D coordinates (x) of each atom in the molecule
        one_hot (torch.Tensor): [n, num_atom_classes] giving the one_hot encoded classes of each atom in the molecule
        charges (torch.Tensor): [n, 1] giving the number of protons in each atom in the molecule
        size_log_prob (float): prior log probability of having a molecule of this size
    """
    n_nodes: int
    coords: torch.Tensor
    one_hot: torch.Tensor
    charges: torch.Tensor
    size_log_prob: float

@dataclass
class EDMDataloaderItem:
    """
    A class representing a batch of molecules. Note that we use "flattened" notation, in that coords and features are one long tensor of size [N, 3] or [N, num_atom_classes] respectively, where N is the number of atoms across the entire batch. This allows each molecule to have a different number of atoms and we keep track of this dynamically.

    We treat each batch as one single large "graph": each individual molecule in the batch has fully connected nodes. Therefore in order to represent the edges in the batch we only require tensors of outer length NN, where NN = (n_nodes ** 2).sum() is the number of edges in the batch.

    Attributes:
        n_nodes (torch.Tensor): [B] where B is the batch size and each bth entry is the number of atoms in the bth molecule. Let N = n_nodes.sum() be the number of atoms in the entire batch and NN = (n_nodes ** 2).sum(), the number of edges in the batch.
        coords (torch.Tensor): [N, 3], where N = n_nodes.sum(), giving the 3D coordinates of each atom in the batch
        one_hot (torch.Tensor): [N, num_atom_classes] giving the class of the each atom in the batch
        charges (torch.Tensor): [N, 1] giving the charge of each atom in the batch
        edges (torch.Tensor): (NN, 2) tensor, where NN = (n_nodes ** 2).sum(). Each row of this tensor gives the indices of atoms within the batch for which there is a directed edge. Note that we do allow edges from a node to itself as this simplifies dealing with the ordering of the edges across the code
        reduce (torch.Tensor): (N, NN) binary float matrix where reduce[i, j] = 1.0 if and only if there is an edge from atom i->j
        batch_mean (torch.Tensor): (B, N) float matrix such that given an [N, dim] matrix A, the operation mean @ A gives an [B, dim] matrix where information is averaged over molecules
        batch_sum (torch.Tensor): (B, N) float matrix such that given an [N, dim] matrix A, the operation mean @ A gives an [B, dim] matrix where information is summed over molecules
        demean (torch.Tensor): (NN, NN) float matrix such that given an [NN, dim] matrix A, the operatation demean @ A gives an [NN, dim] matrix where molecule-wise centre of masses are zero
        expand_idx (torch.Tensor): (N) if a is a vector of length B with elements corresponding to each molecule in the batch, then a[expand_idx] is a vector of length N where each element is repeated for every atom in the molecule
        size_log_probs (torch.Tensor): [B] prior log probs of molecule size
    """
    n_nodes: torch.Tensor
    coords: torch.Tensor
    one_hot: torch.Tensor
    charges: torch.Tensor
    edges: torch.Tensor
    reduce: torch.Tensor
    batch_mean: torch.Tensor
    batch_sum: torch.Tensor
    demean: torch.Tensor
    expand_idx: torch.Tensor
    size_log_probs: torch.Tensor
    
    def to_(self, *args, **kwargs):
        """calls `torch.Tensor.to` on all tensors, assigning the results
        """
        self.n_nodes = self.n_nodes.to(*args, **kwargs)
        self.coords = self.coords.to(*args, **kwargs)
        self.one_hot = self.one_hot.to(*args, **kwargs)
        self.charges = self.charges.to(*args, **kwargs)
        self.edges = self.edges.to(*args, **kwargs)
        self.reduce = self.reduce.to(*args, **kwargs)
        self.batch_mean = self.batch_mean.to(*args, **kwargs)
        self.batch_sum = self.batch_sum.to(*args, **kwargs)
        self.demean = self.demean.to(*args, **kwargs)
        self.expand_idx = self.expand_idx.to(*args, **kwargs)
        self.size_log_probs = self.size_log_probs.to(*args, **kwargs)

QM9Attributes = Literal["index", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", "omega1", "zpve_thermo", "U0_thermo", "U_thermo", "H_thermo", "G_thermo", "Cv_thermo"]
QM9ProcessedData = dict[Literal["num_atoms", "classes", "charges", "positions", QM9Attributes], torch.Tensor]

class EDMDataset(ABC, td.Dataset):
    def __init__(self, num_atom_classes, max_nodes, size_histogram):
        """initialise EDM dataset

        Args:
            num_atom_classes (int): number of data features
            max_nodes (int): maximum number of atoms per molecule
            size_histogram (dict[int, int]|None): count of molecule sizes
        """
        super().__init__()
        assert num_atom_classes > 2

        self.num_atom_classes = num_atom_classes
        self.max_nodes = max_nodes
        self.size_histogram = size_histogram
        
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
        super().__init__(num_atom_classes=QM9["num_atom_classes"] if use_h else QM9_NO_H["num_atom_classes"], max_nodes=QM9["largest_molecule_size"] if use_h else QM9_NO_H["largest_molecule_size"], size_histogram=QM9["molecule_size_histogram"] if use_h else QM9_NO_H["molecule_size_histogram"])
        
        assert split in ["train", "valid", "test"]
        
        self.split = split
        self.use_h = use_h
        
        processed_filename = f"{split}_{"h" if use_h else "no_h"}"
        self._processed_path = path.join(args.data_dir, "qm9", f"{processed_filename}.npz")
        if not (Path(self._processed_path).is_file() and check_hash_file(self._processed_path, getattr(args, f"qm9_{processed_filename}_npz_md5"))):
            ensure_qm9_processed(path.join(args.data_dir, "qm9"), use_h)
        
        self._data: QM9ProcessedData = {key: torch.from_numpy(val) for key, val in np.load(self._processed_path).items()}
        self._len  = len(self._data["num_atoms"])

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index) -> EDMDatasetItem:
        n_nodes = self._data["num_atoms"][index]
        coords = self._data["positions"][index, :n_nodes]
        classes = self._data["classes"][index, :n_nodes]
        charges = self._data["charges"][index, :n_nodes]
        eye = torch.eye(self.num_atom_classes)
        size_log_prob = self.log_size_histogram_probs[int(n_nodes)]
        
        return EDMDatasetItem(
            n_nodes = int(n_nodes),
            coords = coords,
            charges = charges,
            one_hot = eye[classes],
            size_log_prob = size_log_prob
        )

class DummyDataset(EDMDataset):
    """Dummy dataset
    """
    def __init__(self, len=10000, num_atom_classes=7, max_nodes=25):
        super().__init__(num_atom_classes, max_nodes, {n: 1 for n in range(1, max_nodes + 1)})
        self.len = len
        self.rng = torch.random

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> EDMDatasetItem:
        """
        Args:
            index (int): the index

        Returns:
            dict[str, Any]: returns a dict containing "coords" and "n_nodes"
        """
        n_nodes = int(torch.randint(low=2, high=self.max_nodes + 1, size=(), dtype=torch.int64))
        coords = torch.randn((n_nodes, 3), dtype=torch.float32)
        charges = 1 + torch.randint(low=0, high=self.num_atom_classes, size=(n_nodes,), dtype=torch.int64)
        one_hot = torch.eye(self.num_atom_classes, dtype=torch.float32)[charges - 1]
        size_log_prob = - log(self.max_nodes)
        

        return EDMDatasetItem(n_nodes=n_nodes, coords=coords, one_hot=one_hot, charges=charges, size_log_prob=size_log_prob)

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

        return EDMDataloaderItem(n_nodes=n_nodes, coords=coords, one_hot=one_hot, charges=charges, edges=edges, reduce=reduce, batch_mean=batch_mean, batch_sum=batch_sum, demean=demean, expand_idx=expand_idx, size_log_probs=size_log_probs)

def get_dummy_dataloader(num_atom_classes: int, len: int, max_nodes: int, batch_size: int):
    return td.DataLoader(dataset=DummyDataset(num_atom_classes=num_atom_classes, max_nodes=max_nodes, len=len), batch_size=batch_size, collate_fn=_collate_fn)

def get_qm9_dataloader(use_h: bool, split: Literal["train", "valid", "test"], batch_size: int, prefetch_factor: int|None=4, num_workers = 0 if cpu_count() < 4 else int(0.5 * cpu_count()), pin_memory=True, shuffle=True):
    return td.DataLoader(dataset=QM9Dataset(use_h=use_h, split=split), batch_size=batch_size, collate_fn=_collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=num_workers, shuffle=shuffle)