from typing import Any
import torch
from torch import nn
from dataclasses import dataclass
from data import EDMDataloaderItem

@dataclass
class EGCLConfig:
    num_layers: int
    hidden_d: int
    features_d: int
    node_attr_d: int
    edge_attr_d: int
    use_tanh: bool
    tanh_range: float

@dataclass
class EGNNConfig(EGCLConfig):
    num_layers: int
    use_resid: bool
    
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
            nn.Linear(config.features_d   +   config.features_d   +   1   +   config.edge_attr_d, config.hidden_d),
            nn.SiLU(),
            nn.Linear(config.hidden_d, config.hidden_d),
            nn.SiLU()
        )

        coord_final_layer = nn.Linear(config.hidden_d, 1, bias=False)
        # make the final layer of the coord_mlp have small initial weights, to avoid the network blowing up when we start training (empirically if we do not include this line, then every layer of the EGCL gives a larger magnitude for coords and features, so by the last layer we have a tensor full of nans)
        nn.init.xavier_uniform_(coord_final_layer.weight, gain=0.001)
        # the \(\phi_{x}\) network for coordinate update (same architecture as above but we have an additional projection onto a scalar)
        self.coord_mlp = nn.Sequential(
            #            hi               hj            dij2         aij
            nn.Linear(config.features_d   +   config.features_d   +   1   +   config.edge_attr_d, config.hidden_d),
            nn.SiLU(),
            nn.Linear(config.hidden_d, config.hidden_d),
            nn.SiLU(),
            coord_final_layer,
        )

        # the \(\phi_{h}\) network for node update
        self.node_mlp = nn.Sequential(
            nn.Linear(config.features_d + config.hidden_d + config.node_attr_d, config.hidden_d),
            nn.SiLU(),
            nn.Linear(config.hidden_d, config.features_d)
        )

        # the \(\phi_{inf}\) network for edge inference operator
        self.inf_mlp = nn.Sequential(
            nn.Linear(config.hidden_d, 1),
            nn.Sigmoid()
        )
        
        for p in self.modules():
            fan_out_init(p, skip=[self.coord_mlp[-1]])

        # save things
        self.config = config

    def forward(self, coords: torch.Tensor, features: torch.Tensor, edges: torch.Tensor, reduce: torch.Tensor, node_attr: torch.Tensor|None=None, edge_attr: torch.Tensor|None=None):
        """forward pass for equivariant graph convolutional layer

        Args:
            coords (torch.Tensor): [N, 3]
            features (torch.Tensor): [N, features_d]
            edges (torch.Tensor): [NN, 2] (torch.int64)
            reduce (torch.Tensor): [N, NN] (tensor of 1.0's and 0.0's that is a float)
            node_attr (torch.Tensor | None, optional): Node attributes. None (if using no node attributes), or [N, node_attr_d]. Defaults to None.
            edge_attr (torch.Tensor | None, optional): Edge attributes. None (if using no edge attributes), or [NN, node_attr_d]. Defaults to None.
        """

        x_e = coords[edges]  # [NN, 2, 3]
        h_e = features[edges]  # [NN, 2, features_d]
        h_e_flattened = h_e.flatten(start_dim=1)  # [NN, 2 * features_d]
        d_e = (x_e[:, 0, :] - x_e[:, 1, :]).norm(dim=1).unsqueeze(dim=1) # [NN, 1]

        # calculate output for the \(\phi_{e}\) and \(\phi_{x}\) networks
        coord_mlp_in = edge_mlp_in = torch.cat([h_e_flattened, d_e ** 2] + ([edge_attr] if edge_attr is not None else []), dim=-1)
        phi_e = self.edge_mlp(edge_mlp_in)  # [NN, hidden_d]
        phi_x = self.coord_mlp(coord_mlp_in)   # [NN, 1]

        # calculate output for the \(\phi_{h}\) network, giving feature vector output
        node_mlp_in = torch.cat([features, reduce @ (phi_e * self.inf_mlp(phi_e))] + ([node_attr] if node_attr is not None else []), dim=-1)
        features_out: torch.Tensor = features + self.node_mlp(node_mlp_in)  # [N, features_d]

        # calculate coord vector output
        # w = phi_x / (d_e + 1)  # [NN, 1]
        if self.config.use_tanh:
            w = self.config.tanh_range * torch.tanh(phi_x)
        else:
            w = phi_x
            
        normalised_coord_differences = (x_e[:, 0, :] - x_e[:, 1, :]) / (torch.clamp(d_e, min=1e-5) + 1)
        coords_out: torch.Tensor = coords + reduce @ (w * normalised_coord_differences)  # [N, 3]

        return coords_out, features_out



class EGNN(nn.Module):
    def __init__(self, config: EGNNConfig):
        super().__init__()
        assert config.num_layers > 1

        self.embedding  = nn.Linear(
            in_features=config.features_d + 1,  # t/T is an additional non-noised feature
            out_features=config.hidden_d,
        )
        self.embedding_out = nn.Linear(
            in_features=config.hidden_d,
            out_features=config.features_d + 1,  # again, t/T is an additional non-noised feature
        )
        
        self.egc_layers = nn.ModuleList([
            # note: all layers except the first have an additional edge attribute that is equal to the squared coordinate distance at the first layer (page 13)
            EGCL(EGCLConfig(
                features_d=config.hidden_d,  # features are already projected into latent space
                num_layers=config.num_layers,
                hidden_d=256,  # features are already projected into latent space
                node_attr_d=config.node_attr_d,
                edge_attr_d=config.edge_attr_d + 1, # distance between atoms at layer 0 becomes an additional extra edge attribute for layers 0, 1, 2, ... note that there is redundancy at layer 0 but we ignore this
                use_tanh=config.use_tanh,
                tanh_range=config.tanh_range)) for l in range(config.num_layers)
        ])

        self.config = config

    def forward(self, n_nodes: torch.Tensor, coords: torch.Tensor, features: torch.Tensor, edges: torch.Tensor, reduce: torch.Tensor, demean: torch.Tensor, time: float | torch.Tensor, node_attr: torch.Tensor|None=None, edge_attr: torch.Tensor|None=None):
        x_e = coords[edges]  # [NN, 2, 3]
        d_e = (x_e[:, 0, :] - x_e[:, 1, :]).norm(dim=1).unsqueeze(dim=1) # [NN, 1]

        coords_out = coords
        
        if isinstance(time, float):
            time = time * torch.ones(size=(coords.shape[0], 1), dtype=coords.dtype, layout=coords.layout, device=coords.device)            
        assert isinstance(time, torch.Tensor)        
        if len(time.shape) == 1:
            time = time[:, None]
            
        hidden = self.embedding(torch.cat([features, time], dim=-1))  # put noised features into latent space; t/T is an additional non-noised feature
        
        edge_attr_with_distance = torch.cat([d_e ** 2] + ([edge_attr] if edge_attr is not None else []), dim=-1)

        for l, egcl in enumerate(self.egc_layers):
            # note: all layers except the first have an additional edge attribute that is equal to the squared coordinate distance at the first layer (page 13)
            coords_out, hidden = egcl(coords_out, hidden, edges, reduce, node_attr, edge_attr_with_distance)
            
        features_out = self.embedding_out(hidden)[:, :-1]  # put hidden back into ambient space, and cut off the final bit representing t/T
        coords_out = demean @ coords_out
        if self.config.use_resid:
            coords_out = coords_out - coords

        return coords_out, features_out
