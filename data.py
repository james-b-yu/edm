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
from utils.files import hash_file
from utils.qm9 import charge_to_idx, ensure_qm9_raw_data, ensure_qm9_raw_excluded, ensure_qm9_raw_splits, ensure_qm9_raw_thermo, ensure_qm9_processed

@dataclass
class EDMDatasetItem:
    """Attributes:
        n_nodes (int): n, number of atoms in the molecule
        coords (torch.Tensor): [n, 3] giving the 3D coordinates (x) of each atom in the molecule
        features (torch.Tensor): [n, num_atom_classes] giving the features (h) of each atom in the molecule
    """
    n_nodes: int
    coords: torch.Tensor
    features: torch.Tensor

@dataclass
class EDMDataloaderItem:
    """
    A class representing a batch of molecules. Note that we use "flattened" notation, in that coords and features are one long tensor of size [N, 3] or [N, num_atom_classes] respectively, where N is the number of atoms across the entire batch. This allows each molecule to have a different number of atoms and we keep track of this dynamically.

    We treat each batch as one single large "graph": each individual molecule in the batch has fully connected nodes. Therefore in order to represent the edges in the batch we only require tensors of outer length NN, where NN = (n_nodes ** 2).sum() is the number of edges in the batch.

    Attributes:
        n_nodes (torch.Tensor): [B] where B is the batch size and each bth entry is the number of atoms in the bth molecule. Let N = n_nodes.sum() be the number of atoms in the entire batch and NN = (n_nodes ** 2).sum(), the number of edges in the batch.
        coords (torch.Tensor): [N, 3], where N = n_nodes.sum(), giving the 3D coordinates of each atom in the batch
        featuers (torch.Tensor): [N, num_atom_classes] giving the features of the each atom in the batch
        edges (torch.Tensor): (NN, 2) tensor, where NN = (n_nodes ** 2).sum(). Each row of this tensor gives the indices of atoms within the batch for which there is a directed edge. Note that we do allow edges from a node to itself as this simplifies dealing with the ordering of the edges across the code
        reduce (torch.Tensor): (N, NN) binary float matrix where reduce[i, j] = 1.0 if and only if there is an edge from atom i->j
        demean (torch.Tensor): (NN, NN) float matrix such that given an [NN, dim] matrix A, the operatation demean @ A gives an [NN, dim] matrix where molecule-wise centre of masses are zero
    """
    n_nodes: torch.Tensor
    coords: torch.Tensor
    features: torch.Tensor
    edges: torch.Tensor
    reduce: torch.Tensor
    demean: torch.Tensor
    
    def to_(self, *args, **kwargs):
        """calls `torch.Tensor.to` on all tensors, assigning the results
        """
        self.n_nodes = self.n_nodes.to(*args, **kwargs)
        self.coords = self.coords.to(*args, **kwargs)
        self.features = self.features.to(*args, **kwargs)
        self.edges = self.edges.to(*args, **kwargs)
        self.reduce = self.reduce.to(*args, **kwargs)
        self.demean = self.demean.to(*args, **kwargs)

QM9Attributes = Literal["index", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", "omega1"]
QM9ProcessedData = dict[Literal["num_atoms", "classes", "charges", "positions", QM9Attributes], torch.Tensor]

class EDMDataset(ABC, td.Dataset):
    def __init__(self, num_atom_classes, max_nodes):
        """initialise EDM dataset

        Args:
            num_atom_classes (int, optional): number of data features. Defaults to 7.
            max_nodes (int, optional): maximum number of atoms per molecule. Defaults to 25.
        """
        super().__init__()
        assert num_atom_classes > 2

        self.num_atom_classes = num_atom_classes
        self.max_nodes = max_nodes

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> EDMDatasetItem:
        pass

class QM9Dataset(EDMDataset):
    def __init__(self, use_h: bool, split: Literal["train", "valid", "test"]):
        super().__init__(num_atom_classes=5 if use_h else 4, max_nodes=29 if use_h else 9) # TODO
        
        assert split in ["train", "valid", "test"]
        
        self.split = split
        self.use_h = use_h
        
        processed_filename = f"{split}_{"h" if use_h else "no_h"}"
        self._processed_path = path.join(args.data_dir, "qm9", f"{processed_filename}.npz")
        if not (Path(self._processed_path).is_file() and hash_file(self._processed_path) == getattr(args, f"qm9_{processed_filename}_npz_md5")):
            ensure_qm9_processed(path.join(args.data_dir, "qm9"), use_h)
        
        self._data: QM9ProcessedData = {key: torch.from_numpy(val) for key, val in np.load(self._processed_path).items()}
        self._len  = len(self._data["num_atoms"])

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index) -> EDMDatasetItem:
        n_nodes = self._data["num_atoms"][index]
        coords = self._data["positions"][index, :n_nodes]
        classes = self._data["classes"][index, :n_nodes]
        eye = torch.eye(self.num_atom_classes)
        
        return EDMDatasetItem(
            n_nodes = int(n_nodes),
            coords = coords,
            features = eye[classes]
        )

class DummyDataset(EDMDataset):
    """Dummy dataset
    """
    def __init__(self, len=10000, num_atom_classes=7, max_nodes=25):
        super().__init__(num_atom_classes, max_nodes)
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
        features = torch.eye(self.num_atom_classes, dtype=torch.float32)[torch.randint(low=0, high=self.num_atom_classes, size=(n_nodes,), dtype=torch.int64)]

        return EDMDatasetItem(n_nodes=n_nodes, coords=coords, features=features)

def _collate_fn(data: list[EDMDatasetItem]):
        n_nodes = torch.tensor([d.n_nodes for d in data], dtype=torch.int64)
        coords = torch.cat([d.coords for d in data])
        features = torch.cat([d.features for d in data])

        _n_nodes_cumsum = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), n_nodes]), dim=0)

        edges = torch.cat([torch.cartesian_prod(torch.arange(b_n.item(), dtype=torch.int64), torch.arange(b_n.item(), dtype=torch.int64)) + _n_nodes_cumsum[b] for b, b_n in enumerate(n_nodes)], dim=0)
        reduce = torch.block_diag(*[torch.block_diag(*[torch.ones((int(n), )).scatter_(0, torch.tensor([m]), 0) for m in range(n)]) for n in n_nodes])
        demean = torch.block_diag(*[torch.eye(int(n), dtype=torch.float32) - (torch.ones((int(n), int(n)), dtype=torch.float32) / n) for n in n_nodes])

        coords = demean @ coords  # immediately demean the coords

        return EDMDataloaderItem(n_nodes=n_nodes, coords=coords, features=features, edges=edges, reduce=reduce, demean=demean)

def get_dummy_dataloader(num_atom_classes: int, len: int, max_nodes: int, batch_size: int):
    return td.DataLoader(dataset=DummyDataset(num_atom_classes=num_atom_classes, max_nodes=max_nodes, len=len), batch_size=batch_size, collate_fn=_collate_fn)

def get_qm9_dataloader(use_h: bool, split: Literal["train", "valid", "test"], batch_size: int, prefetch_factor: int|None=4, num_workers = 0 if cpu_count() < 4 else int(0.5 * cpu_count()), pin_memory=True, shuffle=True):
    return td.DataLoader(dataset=QM9Dataset(use_h=use_h, split=split), batch_size=batch_size, collate_fn=_collate_fn, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=num_workers, shuffle=shuffle)