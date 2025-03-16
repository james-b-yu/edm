if False:
    """
    we no longer want to use the masked model
    """
    from argparse import Namespace
    from dataclasses import dataclass
    from typing import Literal
    import torch
    from torch import nn

    from model import BaseEDM
    from model_config import EDMConfig, EGCLConfig, EGNNConfig
    from utils.diffusion import cosine_beta_schedule, demean_using_mask, get_batch_edge_idx, get_coord_distance, polynomial_schedule, unsorted_segment_sum

    class MaskedEGCL(nn.Module):
        def __init__(self, config: EGCLConfig):
            super().__init__()
            self.config = config

            # the \(\phi_{e}\) network for edge operation
            self.edge_mlp = nn.Sequential(
                #            hi               hj            dij2       aij
                nn.Linear(config.hidden_dim   +   config.hidden_dim   +   2, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU()
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
            # the \(\phi_{x}\) network for coordinate update (same architecture as above but we have an additional projection onto a scalar)
            coord_mlp_final_layer = nn.Linear(config.hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(coord_mlp_final_layer.weight, gain=0.001)
            self.coord_mlp = nn.Sequential(
                #            hi               hj            dij2       aij
                nn.Linear(config.hidden_dim   +   config.hidden_dim   +   2, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                coord_mlp_final_layer)
        
        def op_edge(self, feat, sq_dists, edge_idx, edge_mask):
            r, c = edge_idx
            input = torch.cat([feat[r], feat[c], sq_dists], dim=-1)
            output = self.edge_mlp(input)
            attention = self.inf_mlp(output)
            return output * attention * edge_mask
        
        def op_node(self, feat, phi_e, edge_idx, node_mask):
            r, _ = edge_idx
            sums = unsorted_segment_sum(phi_e, r, feat.shape[0])
            input = torch.cat([feat, sums], dim=-1)
            output = feat + self.node_mlp(input)
            return output * node_mask

        def op_coord(self, coord, phi_h, norm_coord_diff, sq_dists, edge_idx, edge_mask):
            r, c = edge_idx
            input = torch.cat([phi_h[r], phi_h[c], sq_dists], dim=-1)
            output = norm_coord_diff * self.coord_mlp(input).tanh() * self.config.tanh_multiplier * edge_mask
            sums = unsorted_segment_sum(output, r, num_segments=coord.shape[0])
            return coord + sums
            
        def forward(self, coord: torch.Tensor, feat: torch.Tensor, edge_idx: tuple[torch.Tensor, torch.Tensor], node_mask: torch.Tensor, edge_mask: torch.Tensor, squared_z_coord_distances: torch.Tensor):
            sq_layer_coord_dist, norm_coord_diff = get_coord_distance(coord, edge_idx)
            sq_dists = torch.cat([sq_layer_coord_dist, squared_z_coord_distances], dim=-1)  # [B*N, 2] tensor where the first column is the squared distance in current layer representations of z, and the second column is the squared distance in z input; if this is the first layer, then the first and second columns are the same
            phi_e = self.op_edge(feat=feat, sq_dists=sq_dists, edge_idx=edge_idx, edge_mask=edge_mask)
            phi_h = self.op_node(feat=feat, phi_e=phi_e, edge_idx=edge_idx, node_mask=node_mask)
            phi_x = self.op_coord(coord=coord, phi_h=phi_h, norm_coord_diff=norm_coord_diff, sq_dists=sq_dists, edge_idx=edge_idx, edge_mask=edge_mask)
            return phi_x, phi_h

    class MaskedEGNN(nn.Module):
        def __init__(self, config: EGNNConfig):
            super().__init__()
            self.config = config

            self.embedding = nn.Linear(config.num_atom_types + 2, config.hidden_dim)  # features are cat(one_hot, charge, t/T), so we add 2 additional dims
            self.embedding_out = nn.Linear(config.hidden_dim, config.num_atom_types + 2)
            self.egcls = nn.ModuleList([
                MaskedEGCL(config) for l in range(config.num_layers)
            ])
        
        def forward(self, z_coord: torch.Tensor, z_feat: torch.Tensor, time_frac: float|torch.Tensor, node_mask: torch.Tensor, edge_mask: torch.Tensor):
            batch_size, max_num_atoms, _ = z_coord.shape
            contig_size = batch_size * max_num_atoms
            _, _, dim_feat = z_feat.shape
            # flatten data so that atoms are continguous
            node_mask = node_mask.flatten()[:, None]
            edge_mask = edge_mask.flatten()[:, None]
            z_coord = z_coord.flatten(start_dim=0, end_dim=1)
            z_coord_in = z_coord.clone()
            z_feat = z_feat.flatten(start_dim=0, end_dim=1)
            # get indices for batch-wise fully connected graph if data is flattend such that atoms are contiguous
            edge_idx = get_batch_edge_idx(max_num_atoms, batch_size, self.config.device)
            # create time tensor by repeating time across all atoms in each molecule
            if isinstance(time_frac, float):  # same time for every molecule
                time_frac = torch.full_like(z_feat[:, 0:1], fill_value=time_frac)
            else:  # different time for each molecule
                assert isinstance(time_frac, torch.Tensor) and time_frac.shape == (batch_size, )
                time_frac = time_frac[:, None].repeat(1, max_num_atoms).flatten()[:, None]
            # finally, concat together features and time fraction. XXX: if we want to do conditional generation, then we add the context here
            z_feat = torch.cat([z_feat, time_frac], dim=-1)
            
            # project features into high-dimensional latent space
            h_feat = self.embedding(z_feat)
            z_squared_coord_dist, _ = get_coord_distance(z_coord, edge_idx)
            
            # propage through equivariant graph convolutional layers
            for l in range(self.config.num_layers):
                z_coord, h_feat = self.egcls[l](coord=z_coord, feat=h_feat, edge_idx=edge_idx, node_mask=node_mask, edge_mask=edge_mask, squared_z_coord_distances=z_squared_coord_dist)
            
            # unflatten predictions for (epsilons of) coordinates and features
            node_mask = node_mask.view(batch_size, max_num_atoms, -1)
            out_coord = z_coord - z_coord_in
            out_coord = out_coord.view(batch_size, max_num_atoms, -1)  # unflatten coordinates
            out_coord = demean_using_mask(out_coord, node_mask)
            
            out_feat = self.embedding_out(h_feat)
            out_feat = out_feat[:, :-1].view(batch_size, max_num_atoms, -1) * node_mask  # we multiply again by node_mask since the bias term of embedding_out may be non-zero. but this is redundant if we never look at values corresponding to padding indices
            return out_coord, out_feat

    class MaskedEDM(BaseEDM):
        """this is our masked implemention of vanilla EDM
        """
        def __init__(self, config: EDMConfig):
            super().__init__(config)
            self.egnn = MaskedEGNN(config)
            self.to(device=config.device)
            
        def get_eps_and_predicted_eps(self, coord: torch.Tensor, one_hot: torch.Tensor, charge: torch.Tensor, time_int: torch.Tensor | int, node_mask: torch.Tensor, edge_mask: torch.Tensor):
            """
            this method is useful during training and evaluation
            given ground truth data, and (potentially random) time generate epsilons and perform one evaluation of the model to predict these epsilons

            Args:
                coord (torch.Tensor): self-explanatory
                one_hot (torch.Tensor): self-explanatory
                charge (torch.Tensor): self-explanatory
                time_int (torch.Tensor | int): either an integer, in which case all molecules in the batch will have the same time step, or a long-tensor of shape [batch_size]. note this is an INTEGER from 0 to num_steps, inclusive
                node_mask (torch.Tensor): get these from the dataloader
                edge_mask (torch.Tensor): get these from the dataloader

            Returns:
                tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: actual and predicted epsilons, ready to calculate squared difference norm or mse, etc.
            """
            s_coord, s_one_hot, s_charge = self.scale_inputs(coord, one_hot, charge)
            if isinstance(time_int, float):
                time_int = torch.full(fill_value=time_int, size=(s_coord.shape[0], ), dtype=torch.long)
            assert(isinstance(time_int, torch.Tensor) and time_int.dtype == torch.long)
            
            alf = self.schedule["alpha"][time_int][:, None, None]
            sig = self.schedule["sigma"][time_int][:, None, None]
            
            s_feat = torch.cat([s_one_hot, s_charge], dim=-1)
            
            eps_feat = torch.randn_like(s_feat) * node_mask
            
            x_masked = torch.randn_like(s_coord) * node_mask
            eps_coord   = demean_using_mask(x_masked, node_mask)

            z_coord = alf * s_coord + sig * eps_coord
            z_feat  = alf * s_feat + sig * eps_feat
            
            time_frac = time_int / self.config.num_steps
            (pred_eps_coord, pred_eps_feat) = self.egnn(z_coord, z_feat, time_frac, node_mask, edge_mask)
        
            return (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat)