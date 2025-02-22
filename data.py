import torch
from torch.utils import data as td

class DummyDataset(td.Dataset):
    """Dummy dataset
    """
    def __init__(self, features_d=7, len=10000, max_nodes=25):
        """initialise dummy dataset

        Args:
            features_d (int, optional): number of data features. Defaults to 7.
            len (int, optional): dummy dataset length. Defaults to 10000.
            max_nodes (int, optional): maximum number of atoms per molecule. Defaults to 25.
        """
        super().__init__()
        self.features_d = features_d
        self.len = len
        self.max_nodes = max_nodes
        self.rng = torch.random
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        """
        Args:
            index (int): the index

        Returns:
            dict[str, Any]: returns a dict containing "coords" and "n_nodes"
        """
        n_nodes = int(torch.randint(low=1, high=self.max_nodes + 1, size=(), dtype=torch.int64))
        coords = torch.randn((n_nodes, 3), dtype=torch.float64)
        features = torch.randn((n_nodes, self.features_d), dtype=torch.float64)
        
        if index == 0:
            n_nodes = 2
            coords = torch.tensor([[0, 0, 0],
                                    [1, 1, 1]])
        else:
            n_nodes = 3
            coords = torch.tensor([[0, 0, 0],
                                    [1, 1, 1],
                                    [2, 2, 2]])
        
        return {
            "coords": coords,
            "features": features,
            "n_nodes": n_nodes
        }
        
        
        
        
        
def get_dummy_dataloader(features_d: int, len: int, max_nodes: int, batch_size: int, device, use_sparse=True):
    def collate_fn(data):
        n_nodes = torch.tensor([d["n_nodes"] for d in data], dtype=torch.int64)  # (B, ) tensor where each bth entry is the number of atoms in the bth molecule
        coords = torch.cat([d["coords"] for d in data])  # (N, 3) tensor, where N = n_nodes.sum(), and each nth entry is the 3D coordinate of the nth atom in the batch
        features = torch.cat([d["features"] for d in data])  # (N, features_d) tensor, where N = n_nodes.sum(), and each nth entry is the 3D coordinate of the nth atom in the batch
        
        _n_nodes_cumsum = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), n_nodes]), dim=0)
        edges = torch.cat([torch.cartesian_prod(torch.arange(b_n.item(), dtype=torch.int64), torch.arange(b_n.item(), dtype=torch.int64)) + _n_nodes_cumsum[b] for b, b_n in enumerate(n_nodes)], dim=0)  # (NN, 2) tensor, where NN = (n_nodes ** 2).sum(). Each row of this tensor gives the indices of atoms within the batch for which there is a directed edge. Note that we do allow edges from a node to itself as this simplifies the ordering of the edges. (Otherwise when dealing with indices of edges, we must keep track of which pairs are from an atom to itself which is more complicated)
        reduce = torch.block_diag(*[torch.block_diag(*[torch.ones((int(n), )).scatter_(0, torch.tensor([m]), 0) for m in range(n)]) for n in n_nodes])  # (N, NN) binary matrix where reduce[i, j] = 1 if and only if there is an edge from atom i->j
        
        # get them all on the relevant device
        n_nodes = n_nodes.to(device)
        coords  = coords.to(device)
        features = features.to(device)
        edges = edges.to(device)
        reduce: torch.Tensor = reduce.to(device)
        
        if use_sparse:
            reduce = reduce.to_sparse_csr()
        
        _xe = coords[edges]
        
        return {
            "coords": coords,
            "features": features,
            "n_nodes": n_nodes,
            "edges": edges,
            "reduce": reduce
        }
    
    return td.DataLoader(dataset=DummyDataset(features_d, len, max_nodes), batch_size=batch_size, collate_fn=collate_fn)