from argparse import Namespace
from math import log
from typing import Literal
import torch
from torch import nn
from data import EDMDataloaderItem
from model import EGNN, EGNNConfig
from utils.diffusion import cdf_standard_gaussian, cosine_beta_schedule, polynomial_schedule, gaussian_KL, gaussian_KL_batch, scale_features, unscale_features

class VarianceDiffusion(nn.Module):
    def __init__(self, egnn_config: EGNNConfig, num_steps: int, schedule: Literal["cosine", "polynomial"], device: torch.device|str):
        super().__init__()
        
        self.egnn = EGNN(egnn_config)
        self.num_steps = num_steps
        self.schedule = cosine_beta_schedule(num_steps, device) if schedule == "cosine" else polynomial_schedule(num_steps, device)
        
        self.V_h = nn.Parameter(0.25 + 0.5 * torch.rand(size=(self.egnn.config.num_atom_types + 2, )))
        self.V_x = nn.Parameter(torch.tensor((1.)))
        
        self.to(device=device)
        
    def gamma_x(self, time: int | torch.Tensor):
        """get generative model backward variance at time t for coords

        Args:
            time (int | torch.Tensor): _description_

        Returns:
            torch.Tensor: [time] vector of variance
        """
        
        res = (self.V_x * self.schedule["beta"][time].log() + (1. - self.V_x) * self.schedule["beta_sigma"][time].log()).exp()
        
        if isinstance(time, int):
            res.squeeze_()
        return res
    
    def gamma_h(self, time: int | torch.Tensor):
        """get generative model backward variance at time t for features

        Args:
            time (int | torch.Tensor): _description_

        Returns:
            torch.Tensor: [time, features_d] matrix of variances
        """
        res = (self.V_h[None, :] * self.schedule["beta"][time, None].log() + (1. - self.V_h[None, :]) * self.schedule["beta_sigma"][time, None].log()).exp()
        
        if isinstance(time, int):
            res.squeeze_()
        return res
        
        
    def forward(self):
        raise NotImplementedError("Please use self.egnn.forward")
    
    def calculate_loss(self, args: Namespace, split: Literal["train", "valid", "test"], data: EDMDataloaderItem, force_t: None|int = None):
        s_coords, s_one_hot, s_charges = data.coords, *scale_features(data.one_hot, data.charges)
        s_features = torch.cat([s_one_hot, s_charges], dim=-1)
        
        # if calculating training loss, treat L0 like L1:T
        if split == "train":
            lowest_t = 0
        else:
            lowest_t = 1
        
        t_batch_int = torch.randint(low=lowest_t, high=args.num_steps + 1, size=(data.n_nodes.shape[0], ), dtype=torch.long, device=args.device)  # [B]
        if force_t is not None:
            t_batch_int.fill_(force_t)
            
        t_nodes_int = t_batch_int[data.expand_idx] # [N]
        s_batch_int = t_batch_int - 1
        s_nodes_int = s_batch_int[data.expand_idx] # [N]
        t_is_zero = (t_batch_int == 0).float()  # Important to compute log p(x | z0).
        
        s = s_nodes_int / args.num_steps
        t = t_nodes_int / args.num_steps
        
        alf_0 = self.schedule["alpha"][0]
        sig_0 = self.schedule["sigma"][0]
        
        alf = self.schedule["alpha"][t_nodes_int]
        sig = self.schedule["sigma"][t_nodes_int]
        alf_lag = self.schedule["alpha"][s_nodes_int]
        sig_lag = self.schedule["sigma"][s_nodes_int]
        
        alf_batch = self.schedule["alpha"][t_batch_int]
        sig_batch = self.schedule["sigma"][t_batch_int]
        alf_lag_batch = self.schedule["alpha"][s_batch_int]
        sig_lag_batch = self.schedule["sigma"][s_batch_int]
        
        sig_sq = self.schedule["sigma_squared"][t_nodes_int]
        
        eps_coords = data.demean @ torch.randn_like(s_coords)
        eps_features = torch.randn_like(s_features)
        
        z_coords = alf[:, None] * s_coords + sig[:, None] * eps_coords
        z_features = alf[:, None] * s_features + sig[:, None] * eps_features
        
        pred_eps_coords, pred_eps_features = self.egnn(n_nodes=data.n_nodes, coords=z_coords, features=z_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=t)
        
        ### TOOLS TO CALCULATE VANILLA LOSS ###
        def get_van_loss_t_greater_than_zero(): # for losses L1, ..., LT
            sq_err_coords = (data.batch_mean @ (eps_coords - pred_eps_coords) ** 2).mean(dim=-1)  # XXX: during training, we take means; during evaluation we take sums
            sq_err_features = (data.batch_mean @ (eps_features - pred_eps_features) ** 2).mean(dim=-1)  # XXX: during training, we take means; during evaluation we take sums
            return 0.5 * (sq_err_coords + sq_err_features)
        
        def get_van_kl_prior(): # for KL between q(zT | x) and standard normal
            alf_T = torch.full(size=(data.coords.shape[0], ), fill_value=float(self.schedule["alpha"][-1]), dtype=data.coords.dtype, device=args.device)  # [N]            
            sig_sq_T = torch.full_like(data.n_nodes, fill_value=float(self.schedule["sigma_squared"][-1]))  # [B]
            
            mu_T_coords = alf_T[:, None] * s_coords 
            mu_T_features = alf_T[:, None] * s_features
            
            kl_features = gaussian_KL_batch(mu_T_features, sig_sq_T, torch.zeros_like(mu_T_features), torch.ones_like(sig_sq_T), n_nodes=data.n_nodes, batch_sum=data.batch_sum, subspace=False)
            kl_coords = gaussian_KL_batch(mu_T_coords, sig_sq_T, torch.zeros_like(mu_T_coords), torch.ones_like(sig_sq_T), n_nodes=data.n_nodes, batch_sum=data.batch_sum, subspace=True)
            return kl_coords + kl_features
            
        def get_van_log_pxh_given_z0_without_constants():
            # for continuous coords, simply use euclidean distance
            log_px_given_z0 = -0.5 * (data.batch_mean @ (eps_coords - pred_eps_coords) ** 2).mean(dim=-1)  # XXX: during training we take means, during inference we take sums
            
            # for integer one_hot and charges, scale back to integer scaling
            z_one_hot = z_features[:, :-1]
            z_charges = z_features[:, -1]
            us_z_one_hot, us_z_charges = unscale_features(z_one_hot, z_charges)
            us_sig_one_hot, us_sig_charges = unscale_features(self.schedule["sigma"][0], self.schedule["sigma"][0])
            
            # for integer-value charges, find gaussian integral of radius 1 around the deviation
            us_charge_err_centered = data.charges - us_z_charges
            
            log_ph_charges_given_z0 = data.batch_sum @ torch.log(
                cdf_standard_gaussian((us_charge_err_centered + 0.5) / us_sig_charges)
                - cdf_standard_gaussian((us_charge_err_centered - 0.5) / us_sig_charges)
                + 1e-10
            ).sum(dim=-1)
            
            # for one-hot values, find gaussian integral around 1, since one-hot encoded
            us_one_hot_err_centered = us_z_one_hot - 1.
            log_ph_one_hot_given_z0_unnormalised = torch.log(
                cdf_standard_gaussian((us_one_hot_err_centered + 0.5) / us_sig_one_hot)
                - cdf_standard_gaussian((us_one_hot_err_centered - 0.5) / us_sig_one_hot)
                + 1e-10
            )
            log_ph_one_hot_given_z0 = data.batch_sum @ (log_ph_one_hot_given_z0_unnormalised - torch.logsumexp(log_ph_one_hot_given_z0_unnormalised, dim=-1, keepdim=True)).sum(dim=-1)
            
            return log_px_given_z0 + log_ph_one_hot_given_z0 + log_ph_charges_given_z0
        
        ### TOOLS TO CALCULATE VANILLA VLB
        def get_van_kl_t_greater_than_zero():
            sq_err_coords = (data.batch_sum @ ((eps_coords - pred_eps_coords) ** 2)).sum(dim=-1)  # XXX: during training, we take means; during evaluation we take sums
            sq_err_features = (data.batch_sum @ ((eps_features - pred_eps_features) ** 2)).sum(dim=-1)  # XXX: during training, we take means; during evaluation we take sums
            weight = 0.5 * ((alf_lag_batch / sig_lag_batch) / (alf_batch / sig_batch) - 1)
            error = weight * (sq_err_coords + sq_err_features)
            return error
        
        def get_van_vlb_zeroth_term():
            # for continuous coords, we resample 
            z0_coords = alf_0 * s_coords + sig_0 * eps_coords
            z0_features = alf_0 * s_features + sig_0 * eps_features
            pred0_eps_coords, pred0_eps_features = self.egnn(n_nodes=data.n_nodes, coords=z0_coords, features=z0_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=0.)
            error0 = -0.5 * (data.batch_sum @ ((eps_coords - pred0_eps_coords) ** 2)).sum(dim=-1)
            log_px_given_z0 = error0  # during inference we take sums
            
            # for integer one_hot and charges, scale back to integer scaling
            z0_one_hot = z0_features[:, :-1]
            z0_charges = z0_features[:, -1]
            us_z0_one_hot, us_z0_charges = unscale_features(z0_one_hot, z0_charges)
            us_sig0_one_hot, us_sig0_charges = unscale_features(sig_0, sig_0)
            
            # for integer-value charges, find gaussian integral of radius 1 around the deviation
            us_charge_err_centered = data.charges - us_z0_charges[:, None]
            
            log_ph_charges_given_z0 = data.batch_sum @ torch.log(
                cdf_standard_gaussian((us_charge_err_centered + 0.5) / us_sig0_charges)
                - cdf_standard_gaussian((us_charge_err_centered - 0.5) / us_sig0_charges)
                + 1e-10
            ).sum(dim=-1)
            
            # for one-hot values, find gaussian integral around 1, since one-hot encoded
            us_one_hot_err_centered = us_z0_one_hot - 1.
            log_ph_one_hot_given_z0_unnormalised = torch.log(
                cdf_standard_gaussian((us_one_hot_err_centered + 0.5) / us_sig0_one_hot)
                - cdf_standard_gaussian((us_one_hot_err_centered - 0.5) / us_sig0_one_hot)
                + 1e-10
            )
            log_one_hot_normalisation_factor = torch.logsumexp(log_ph_one_hot_given_z0_unnormalised, dim=-1, keepdim=True)
            log_one_hot_probabilities = log_ph_one_hot_given_z0_unnormalised - log_one_hot_normalisation_factor
            log_ph_one_hot_given_z0 = data.batch_sum @ (log_one_hot_probabilities * data.one_hot).sum(dim=-1)
            
            return log_px_given_z0 + log_ph_one_hot_given_z0 + log_ph_charges_given_z0
        
        
        if split=="train":
            eps = torch.concat([eps_coords, eps_features], dim=-1)
            pred_eps = torch.concat([pred_eps_coords, pred_eps_features], dim=-1)
            avr_sq_dist = ((eps - pred_eps) ** 2).mean()
            
            loss_t_greater_than_zero = get_van_loss_t_greater_than_zero()  # loss for terms L1, ..., LT
            # loss_term_0 = -get_van_log_pxh_given_z0_without_constants()    # loss for terms L0
            kl_prior = get_van_kl_prior()  # is negligible if we have done things properly
            
            # loss_t = loss_term_0 * t_is_zero + loss_t_greater_than_zero * (1 - t_is_zero)
            loss_t = loss_t_greater_than_zero
            
            loss_training = loss_t + kl_prior - data.size_log_probs
            return loss_training.mean(), avr_sq_dist
        else:
            eps = torch.concat([eps_coords, eps_features], dim=-1)
            pred_eps = torch.concat([pred_eps_coords, pred_eps_features], dim=-1)
            avr_sq_dist = ((eps - pred_eps) ** 2).mean()
            
            kl_t_greater_than_zero = get_van_kl_t_greater_than_zero()
            vlb_zero = -get_van_vlb_zeroth_term()
            const0 = (data.n_nodes - 1.) * 3 * (0.5 * log(2 * torch.pi) + torch.log(sig_0) - torch.log(alf_0)) # = log Z in the paper
            
            vlb_est = const0 + vlb_zero + self.num_steps * kl_t_greater_than_zero
            
            return vlb_est.mean(), avr_sq_dist