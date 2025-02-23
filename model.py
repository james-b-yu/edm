from typing import Any
import torch
from torch import nn
from dataclasses import dataclass
from data import EDMDataloaderItem

@dataclass
class EGCLConfig:
    features_d: int
    node_attr_d: int
    edge_attr_d: int
    hidden_d: int

@dataclass
class EGNNConfig(EGCLConfig):
    num_layers: int

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

        # the \(\phi_{x}\) network for coordinate update (same architecture as above but we have an additional projection onto a scalar)
        self.coord_mlp = nn.Sequential(
            #            hi               hj            dij2         aij
            nn.Linear(config.features_d   +   config.features_d   +   1   +   config.edge_attr_d, config.hidden_d),
            nn.SiLU(),
            nn.Linear(config.hidden_d, config.hidden_d),
            nn.SiLU(),
            nn.Linear(config.hidden_d, 1)
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
        w = phi_x / (d_e + 1)  # [NN, 1]
        coords_out: torch.Tensor = coords + reduce @ (w * (x_e[:, 0, :] - x_e[:, 1, :]))  # [N, 3]

        return coords_out, features_out


