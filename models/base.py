from abc import ABC, abstractmethod
from math import ceil, log
from typing import Any, Literal
from warnings import warn
import torch
from torch import nn
from tqdm import tqdm
from data import EDMDataloaderItem
from model_config import EDMConfig, EGCLConfig, EGNNConfig
from utils.diffusion import cdf_standard_gaussian, cosine_beta_schedule, polynomial_schedule

class EGCL(nn.Module):
    def __init__(self, config: EGCLConfig):
        super().__init__()

        # the \(\phi_{e}\) network for edge operation
        self.edge_mlp = nn.Sequential(
            #            hi               hj            dij2         aij
            nn.Linear(config.hidden_dim   +   config.hidden_dim   +   2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU()
        )

        coord_final_layer = nn.Linear(config.hidden_dim, 1, bias=False)
        # make the final layer of the coord_mlp have small initial weights, to avoid the network blowing up when we start training (empirically if we do not include this line, then every layer of the EGCL gives a larger magnitude for coords and features, so by the last layer we have a tensor full of nans)
        nn.init.xavier_uniform_(coord_final_layer.weight, gain=config.hidden_dim ** -0.5)
        # the \(\phi_{x}\) network for coordinate update (same architecture as above but we have an additional projection onto a scalar)
        self.coord_mlp = nn.Sequential(
            #            hi               hj            dij2         aij
            nn.Linear(config.hidden_dim   +   config.hidden_dim   +   2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            coord_final_layer,
        )

        # the \(\phi_{h}\) network for node update
        self.node_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # the \(\phi_{inf}\) network for edge inference operator
        self.inf_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )

        # save things
        self.config = config

    def forward(self, coords: torch.Tensor, features: torch.Tensor, edges: torch.Tensor, reduce: torch.Tensor, distance: torch.Tensor):
        """forward pass for equivariant graph convolutional layer

        Args:
            coords (torch.Tensor): [N, 3]
            features (torch.Tensor): [N, features_d]
            edges (torch.Tensor): [NN, 2] (torch.long)
            reduce (torch.Tensor): [N, NN] (tensor of 1.0's and 0.0's that is a float)
            distance (torch.Tensor): [NN, node_attr_d] for distances
        """

        x_e = coords[edges]  # [NN, 2, 3]
        h_e = features[edges]  # [NN, 2, features_d]
        h_e_flattened = h_e.flatten(start_dim=1)  # [NN, 2 * features_d]
        d_e = (x_e[:, 0, :] - x_e[:, 1, :]).norm(dim=1, keepdim=True)

        # calculate output for the \(\phi_{e}\) and \(\phi_{x}\) networks
        edge_mlp_in = torch.cat([h_e_flattened, d_e ** 2, distance], dim=-1)
        phi_e = self.edge_mlp(edge_mlp_in)  # [NN, hidden_d]

        # calculate output for the \(\phi_{h}\) network, giving feature vector output
        node_mlp_in = torch.cat([features, reduce @ (phi_e * self.inf_mlp(phi_e))], dim=-1)
        features_out: torch.Tensor = features + self.node_mlp(node_mlp_in)  # [N, features_d]

        # calculate coord vector output
        coord_mlp_in = torch.cat([features_out[edges].flatten(start_dim=1), d_e ** 2, distance], dim=-1)
        phi_x = self.coord_mlp(coord_mlp_in)   # [NN, 1]
        w = self.config.tanh_multiplier * torch.tanh(phi_x)
            
        normalised_coord_differences = (x_e[:, 0, :] - x_e[:, 1, :]) / (torch.clamp(d_e, min=1e-5) + 1)
        coords_out: torch.Tensor = coords + reduce @ (w * normalised_coord_differences)  # [N, 3]

        return coords_out, features_out



class EGNN(nn.Module):
    def __init__(self, config: EGNNConfig):
        super().__init__()
        assert config.num_layers > 1

        self.embedding  = nn.Linear(
            in_features=config.num_atom_types + 2,  # charges and t/T is an additional non-noised feature
            out_features=config.hidden_dim,
        )
        self.embedding_out = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.num_atom_types + 2,  # again, t/T is an additional non-noised feature
        )
        
        self.egcls = nn.ModuleList([
            # note: all layers except the first have an additional edge attribute that is equal to the squared coordinate distance at the first layer (page 13)
            EGCL(config) for l in range(config.num_layers)
        ])

        self.config = config

    def forward(self, n_nodes: torch.Tensor, coords: torch.Tensor, features: torch.Tensor, edges: torch.Tensor, reduce: torch.Tensor, demean: torch.Tensor, time_frac: float | torch.Tensor):
        x_e = coords[edges]  # [NN, 2, 3]
        d_e = (x_e[:, 0, :] - x_e[:, 1, :]).norm(dim=1).unsqueeze(dim=1) # [NN, 1]

        coords_out = coords
        
        if isinstance(time_frac, int):
            warn(f"time_frac should be a floating-point number between 0.0 and 1.0")
            time_frac = float(time_frac)
        if isinstance(time_frac, float):
            time_frac = time_frac * torch.ones(size=(coords.shape[0], 1), dtype=coords.dtype, layout=coords.layout, device=coords.device)            
        assert isinstance(time_frac, torch.Tensor)        
        if len(time_frac.shape) == 1:
            time_frac = time_frac[:, None]
            
        hidden = self.embedding(torch.cat([features, time_frac], dim=-1))  # put noised features into latent space; t/T is an additional non-noised feature
        
        distance = d_e ** 2

        for l, egcl in enumerate(self.egcls):
            # note: all layers except the first have an additional edge attribute that is equal to the squared coordinate distance at the first layer (page 13)
            coords_out, hidden = egcl(coords_out, hidden, edges, reduce, distance)
            
        features_out = self.embedding_out(hidden)[:, :-1]  # put hidden back into ambient space, and cut off the final bit representing t/T
        coords_out = demean @ coords_out
        coords_out = coords_out - coords

        return coords_out, features_out

class BaseEDM(nn.Module, ABC):
    def __init__(self, config: EDMConfig):
        super().__init__()
        self.config = config
        self.schedule = cosine_beta_schedule(config.num_steps, config.device) if config.schedule_type == "cosine" else polynomial_schedule(config.num_steps, config.device)
        self.egnn = EGNN(config)
        self.to(device=config.device)
        
    def scale_inputs(self, coord, one_hot, charge):
        """given inputs, scale them to prepare for inputs according to config
        """
        s_coord, s_one_hot, s_charge = coord * self.config.coord_in_scale, one_hot * self.config.one_hot_in_scale, charge * self.config.charge_in_scale
        return s_coord, s_one_hot, s_charge
    def unscale_inputs(self, s_coord, s_one_hot, s_charge):
        """given scaled values, unscale them to prepare for outputs according to config
        """
        coord, one_hot, charge = s_coord / self.config.coord_in_scale, s_one_hot / self.config.one_hot_in_scale, s_charge / self.config.charge_in_scale
        return coord, one_hot, charge
    
    def sample(self, num_molecules: int, batch_size: int, mol_sizes: torch.Tensor, mol_size_probs: torch.Tensor, to_numpy=True):
        """sample molecules

        Args:
            num_molecules (int): number of molecules to generate
            batch_size (int): size of minibatches to generate in
            mol_sizes (torch.Tensor): list of possible molecule sizes
            mol_size_probs (torch.Tensor): list of probabilities of molecule sizes, for each size in `mol_sizes`
            to_numpy (bool, optional): whether to return numpy arrays instead of torch tensors. Defaults to True.
            
        Returns
            list[tuple[float np.ndarray, long np.ndarray, long np.ndarray]] if to_numpy is True otherwise list[tuple[float torch.Tensor, long torch.Tensor, long torch.Tensor]]. List of tuples of (coords, one_hot, charges)
        """
        
        samples = []
        assert isinstance(mol_sizes, torch.Tensor) and mol_sizes.dtype == torch.long
        assert isinstance(mol_size_probs, torch.Tensor) and mol_size_probs.dtype == torch.float32
        
        with tqdm(leave=False, total=num_molecules, unit="sample") as pbar:
            for i in range(ceil(num_molecules / batch_size)):
                B = min(batch_size, num_molecules - len(samples))
                num_atoms = mol_sizes[torch.multinomial(mol_size_probs, B, replacement=True)]
                coords, one_hot, charges = self._sample_flattened(num_atoms)
                
                # separate the molecules and append to samples
                batch_idx = 0
                flattened_idx = 0
                while batch_idx < B:
                    size = num_atoms[batch_idx]
                    mol_coords = coords[flattened_idx:flattened_idx + size]
                    mol_one_hot = one_hot[flattened_idx:flattened_idx + size]
                    mol_charges = charges[flattened_idx:flattened_idx + size]
                    if to_numpy:
                        mol_coords = mol_coords.numpy(force=True)
                        mol_one_hot = mol_one_hot.numpy(force=True)
                        mol_charges = mol_charges.numpy(force=True)
                        
                    samples.append((mol_coords, mol_one_hot, mol_charges))
                    flattened_idx += size
                    batch_idx += 1
                
                assert(batch_idx == B and flattened_idx == num_atoms.sum())
                pbar.update(len(samples))
            assert(len(samples) == num_molecules)
        
        return samples

    @abstractmethod
    def estimate_vlb(self, data: EDMDataloaderItem, force_t: None|int = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """calculate estimate for vlb and mse, by generating random time steps

        Args:
            data (EDMDataloaderItem): data to use
            force_t (None | int, optional): if an integer, use the same integer time step for all batches
            
        Return types:
            tuple[torch.Tensor, torch.Tensor]: Both entries are zero-dimensional. The first is an estimate of the VLB, the second is the MSE
        """
        pass
    
    @abstractmethod
    def get_mse(self, data: EDMDataloaderItem, force_t: None|int = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """calculate mse, by generating random time steps

        Args:
            data (EDMDataloaderItem): data to use
            force_t (None | int, optional): if an integer, use the same integer time step for all batches
            
        Return types:
            torch.Tensor: Zero-dimensional
        """
        pass
    
    @abstractmethod
    def _sample_flattened(self, num_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a an array of molecule sizes, return a sampled molecules in flattened format

        Args:
            num_atoms (torch.Tensor): [B] tensor (dtype=torch.long)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: coords, one_hot, charges in flattened representation
        """
        pass
    