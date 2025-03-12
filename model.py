from typing import Any
import torch
from torch import nn
from data import EDMDataloaderItem
from model_config import EDMConfig, EGCLConfig, EGNNConfig
from utils.diffusion import cosine_beta_schedule, polynomial_schedule

def fan_out_init(p: nn.Module, skip: list[nn.Module]):
    if isinstance(p, nn.Linear) and p not in skip:
        nn.init.kaiming_uniform_(p.weight, mode="fan_out", nonlinearity="relu")
        p.weight.data *= 0.01
        if p.bias is not None:
            nn.init.zeros_(p.bias)

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
        nn.init.xavier_uniform_(coord_final_layer.weight, gain=0.001)
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
        
        for p in self.modules():
            fan_out_init(p, skip=[self.coord_mlp[-1]])

        # save things
        self.config = config

    def forward(self, coords: torch.Tensor, features: torch.Tensor, edges: torch.Tensor, reduce: torch.Tensor, distance: torch.Tensor):
        """forward pass for equivariant graph convolutional layer

        Args:
            coords (torch.Tensor): [N, 3]
            features (torch.Tensor): [N, features_d]
            edges (torch.Tensor): [NN, 2] (torch.int64)
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

class BaseEDM(nn.Module):
    def __init__(self, config: EDMConfig):
        super().__init__()
        self.config = config
        self.schedule = cosine_beta_schedule(config.num_steps, config.device) if config.schedule_type == "cosine" else polynomial_schedule(config.num_steps, config.device)
        self.to(config.device)
        
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
    

class EDM(BaseEDM):
    def __init__(self, config: EDMConfig):
        super().__init__(config)
        self.egnn = EGNN(config)
        self.to(config.device)
        
    def get_eps_and_predicted_eps(self, data: EDMDataloaderItem, time_int: torch.Tensor | int):
        """
        this method is useful during training and evaluation
        given ground truth data, and (potentially random) time generate epsilons and perform one evaluation of the model to predict these epsilons

        Args:
            coord (torch.Tensor): self-explanatory
            one_hot (torch.Tensor): self-explanatory
            charge (torch.Tensor): self-explanatory
            time_int (torch.Tensor | int): either an integer, in which case all molecules in the batch will have the same time step, or a long-tensor of shape [batch_size]. note this is an INTEGER from 0 to num_steps, inclusive
            

        Returns:
            tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: actual and predicted epsilons, ready to calculate squared difference norm or mse, etc.
        """
        s_coord, s_one_hot, s_charge = self.scale_inputs(data.coord, data.one_hot, data.charge)
        if isinstance(time_int, float):
            time_int = torch.full(fill_value=time_int, size=(data.batch_size, ), dtype=torch.long)
        assert(isinstance(time_int, torch.Tensor) and time_int.dtype == torch.long and time_int.shape == (data.batch_size, ))
        
        time_int = time_int[data.expand_idx]
        alf = self.schedule["alpha"][time_int][:, None]
        sig = self.schedule["sigma"][time_int][:, None]
        
        s_feat = torch.cat([s_one_hot, s_charge], dim=-1)
        
        eps_feat = torch.randn_like(s_feat)
        eps_coord   = data.demean @ torch.randn_like(s_coord)

        z_coord = alf * s_coord + sig * eps_coord
        z_feat  = alf * s_feat + sig * eps_feat
        
        time_frac = time_int / self.config.num_steps
        (pred_eps_coord, pred_eps_feat) = self.egnn(
            coords=z_coord,
            features=z_feat,
            n_nodes=data.n_nodes,
            edges=data.edges,
            reduce=data.reduce,
            demean=data.demean,
            time_frac=time_frac
        )
    
        return (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat)
