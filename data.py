import torch
from torch.utils import data as td
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

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
        reduce (torch.Tensor): (N, NN) binary matrix where reduce[i, j] = 1 if and only if there is an edge from atom i->j
        demean (torch.Tensor): (NN, NN) float matrix such that given an [NN, dim] matrix A, the operatation demean @ A gives an [NN, dim] matrix where atom-wise centre of masses are zero
    """
    n_nodes: torch.Tensor
    coords: torch.Tensor
    features: torch.Tensor
    edges: torch.Tensor
    reduce: torch.Tensor
    demean: torch.Tensor

class EDMDataset(ABC, td.Dataset):
    def __init__(self, num_atom_classes=7, max_nodes=25):
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

class DummyDataset(EDMDataset):
    """Dummy dataset
    """
    def __init__(self, num_atom_classes=7, max_nodes=25, len=10000):
        super().__init__(num_atom_classes, max_nodes)
        self.len = len
        self.rng = torch.random

    def __len__(self) -> int:
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

def _collate_fn(data: list[EDMDatasetItem], use_sparse: bool, device: torch.device):
        n_nodes = torch.tensor([d.n_nodes for d in data], dtype=torch.int64)
        coords = torch.cat([d.coords for d in data])
        features = torch.cat([d.features for d in data])

        _n_nodes_cumsum = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), n_nodes]), dim=0)

        edges = torch.cat([torch.cartesian_prod(torch.arange(b_n.item(), dtype=torch.int64), torch.arange(b_n.item(), dtype=torch.int64)) + _n_nodes_cumsum[b] for b, b_n in enumerate(n_nodes)], dim=0)
        reduce = torch.block_diag(*[torch.block_diag(*[torch.ones((int(n), )).scatter_(0, torch.tensor([m]), 0) for m in range(n)]) for n in n_nodes])
        demean = torch.block_diag(*[torch.eye(int(n), dtype=torch.float32) - (torch.ones((int(n), int(n)), dtype=torch.float32) / n) for n in n_nodes])

        # get them all on the relevant device
        n_nodes = n_nodes.to(device)
        coords  = coords.to(device)
        features = features.to(device)
        edges = edges.to(device)
        reduce = reduce.to(device)
        demean = demean.to(device)

        coords = demean @ coords  # immediately demean the coords

        if use_sparse:
            reduce = reduce.to_sparse_csr()

        _xe = coords[edges]

        return EDMDataloaderItem(n_nodes=n_nodes, coords=coords, features=features, edges=edges, reduce=reduce, demean=demean)

def get_dummy_dataloader(num_atom_classes: int, len: int, max_nodes: int, batch_size: int, device, use_sparse=True):

    collate_fn = partial(_collate_fn, use_sparse=use_sparse, device=device)

    return td.DataLoader(dataset=DummyDataset(num_atom_classes=num_atom_classes, max_nodes=max_nodes, len=len), batch_size=batch_size, collate_fn=collate_fn)