
from math import log
from warnings import warn
from tqdm import tqdm

import torch
import torch.nn.functional as F
from data import EDMDataloaderItem,get_util_tensors
from configs.model_config import EDMConfig
from models.edm import EDM
from utils.diffusion import cdf_standard_gaussian

from extensions.regularization.disconnection_penalty import get_disconnection_penalty

class RegularizationEDM(EDM):
    """Regularization EDM (penalty for disconnected molecules)
    """
    def __init__(self, config: EDMConfig):
        super().__init__(config)
        self.to(device=config.device)


    def subtract_predicted(self, in_coords, in_features, pred_eps_coords, pred_eps_features, t_nodes_int):
        # Get noise schedule values
        min_eps = 1e-5  # Small constant to prevent division by very small numbers
        alpha_t = torch.clamp(self.schedule.alpha[t_nodes_int], min=min_eps)
        sigma_t = torch.clamp(self.schedule.sigma[t_nodes_int], min=min_eps)

        # Generate random noise and enforce zero center of gravity
        epsilon_x = torch.randn_like(in_coords.T)
        epsilon_x -= epsilon_x.mean(dim=0, keepdim=True)
        epsilon_h = torch.randn_like(in_features.T)
        
        # Apply reverse diffusion update
        coords = ((1 / (alpha_t)) * (in_coords.T - sigma_t * pred_eps_coords.T) + sigma_t * epsilon_x).T
        features = ((1 / (alpha_t)) * (in_features.T - sigma_t * pred_eps_features.T) + sigma_t * epsilon_h).T
        features = torch.cat((F.one_hot(features[:,:5].argmax(dim=-1), num_classes=5).float(), features[:,5:]), dim=1)

        return coords, features
    

    def get_mse(self, data: EDMDataloaderItem, force_t: None|int = None):
        s_coords, s_one_hot, s_charges = self.scale_inputs(data.coords, data.one_hot, data.charges)
        s_features = torch.cat([s_one_hot, s_charges], dim=-1)
        
        lowest_t = 0

        t_batch_int = torch.randint(low=lowest_t, high=self.config.num_steps + 1, size=(data.num_atoms.shape[0], ), dtype=torch.long, device=self.config.device)  # [B]
        if force_t is not None:
            t_batch_int.fill_(force_t)
            
        t_nodes_int = t_batch_int[data.expand_idx] # [N]
        
        t = t_nodes_int / self.config.num_steps
        
        alf_nodes = self.schedule.alpha[t_nodes_int]
        sig_nodes = self.schedule.sigma[t_nodes_int]

        eps_coords = data.demean @ torch.randn_like(s_coords)
        eps_features = torch.randn_like(s_features)
        
        z_coords = alf_nodes[:, None] * s_coords + sig_nodes[:, None] * eps_coords
        z_features = alf_nodes[:, None] * s_features + sig_nodes[:, None] * eps_features
        
        pred_eps_coords, pred_eps_features = self.egnn(n_nodes=data.num_atoms, coords=z_coords, features=z_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=t)
        
        eps = torch.concat([eps_coords, eps_features], dim=-1)
        pred_eps = torch.concat([pred_eps_coords, pred_eps_features], dim=-1)
        avr_sq_dist = ((eps - pred_eps) ** 2).mean()

        post_coords, post_features = self.subtract_predicted(z_coords, z_features, pred_eps_coords, pred_eps_features, t_nodes_int)
        disc_penalty = get_disconnection_penalty(post_coords, post_features, data.num_atoms, self.config.dataset_name, self.config.use_h)
        avr_sq_disc_penalty = (disc_penalty ** 2).mean()

        train_loss = avr_sq_dist + avr_sq_disc_penalty

        if train_loss.isnan():
            warn("Encountered NAN loss.")
            train_loss = torch.full_like(train_loss, fill_value=torch.nan, requires_grad=True)
        return train_loss