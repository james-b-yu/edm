from argparse import Namespace
from math import log
from typing import Literal
from warnings import warn
import torch
from torch import nn
from data import EDMDataloaderItem
from model import EDM, EGNN, EDMConfig
from utils.diffusion import cdf_standard_gaussian, cosine_beta_schedule, polynomial_schedule, gaussian_KL, gaussian_KL_batch, scale_features, unscale_features

class VarianceEDM(EDM):
    def __init__(self, config: EDMConfig):
        super().__init__(config)
        
        self.V_h = nn.Parameter(0.25 + 0.5 * torch.rand(size=(config.num_steps + 1, config.num_atom_types + 1, )))  # [num_steps, num_features] for one_hot and charges and each time step
        self.V_x = nn.Parameter(0.25 + 0.5 * torch.rand(size=(config.num_steps + 1, )))  # [num_steps] for coords
        
        self.to(device=config.device)
        
    def gamma_x(self, time: int | torch.Tensor):
        """get generative model backward variance at time t for coords

        Args:
            time (int | torch.Tensor): integer or long tensor

        Returns:
            torch.Tensor: [time] vector of variance
        """
        
        res = (self.V_x[time] * self.schedule.beta[time].log() + (1. - self.V_x[time]) * self.schedule.rev_beta[time].log()).exp()
        
        if isinstance(time, int):
            res = res.squeeze()
        return res
    
    def gamma_h(self, time: int | torch.Tensor):
        """get generative model backward variance at time t for features

        Args:
            time (int | torch.Tensor): _description_

        Returns:
            torch.Tensor: [time, features_d] matrix of variances
        """
        res = (self.V_h[time] * self.schedule.beta[time, None].log() + (1. - self.V_h[time]) * self.schedule.rev_beta[time, None].log()).exp()
        
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
        
        t_batch_int = torch.randint(low=lowest_t, high=args.num_steps + 1, size=(data.num_atoms.shape[0], ), dtype=torch.long, device=args.device)  # [B]
        if force_t is not None:
            t_batch_int.fill_(force_t)
            
        t_nodes_int = t_batch_int[data.expand_idx] # [N]
        s_batch_int = t_batch_int - 1
        s_nodes_int = s_batch_int[data.expand_idx] # [N]
        
        s = s_nodes_int / args.num_steps
        t = t_nodes_int / args.num_steps
        
        alf_0 = self.schedule.alpha[0]
        sig_0 = self.schedule.sigma[0]
        
        alf_sq_0 = self.schedule.alpha_squared[0]
        sig_sq_0 = self.schedule.sigma_squared[0]
        
        alf_nodes = self.schedule.alpha[t_nodes_int]
        sig_nodes = self.schedule.sigma[t_nodes_int]
        
        alf_batch = self.schedule.alpha[t_batch_int]
        sig_batch = self.schedule.sigma[t_batch_int]
        alf_lag_batch = self.schedule.alpha[s_batch_int]
        sig_lag_batch = self.schedule.sigma[s_batch_int]
        

        alf_sq_batch = self.schedule.alpha_squared[t_batch_int]
        alf_sq_L_batch = self.schedule.alpha_squared_L[t_batch_int]
        sig_sq_batch = self.schedule.sigma_squared[t_batch_int]
        bet_batch = self.schedule.beta[t_batch_int]
        bet_rev_batch = self.schedule.beta[t_batch_int]
        gamma_x_batch = self.gamma_x(t_batch_int)
        gamma_h_batch = self.gamma_h(t_batch_int)
        gamma_h_nodes = self.gamma_h(t_nodes_int)
        
        eps_coords = data.demean @ torch.randn_like(s_coords)
        eps_features = torch.randn_like(s_features)
        
        dim_coords = (data.num_atoms - 1) * 3
        
        z_coords = alf_nodes[:, None] * s_coords + sig_nodes[:, None] * eps_coords
        z_features = alf_nodes[:, None] * s_features + sig_nodes[:, None] * eps_features
        
        pred_eps_coords, pred_eps_features = self.egnn(n_nodes=data.num_atoms, coords=z_coords, features=z_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=t)
        
        gamma_0_coords = self.gamma_x(0)
        gamma_0_feats = self.gamma_h(0)
        
        t_is_zero_float = (t_batch_int == 0).to(dtype=z_coords.dtype)
        
        ### TOOLS TO CALCULATE TRAIN LOSS
        def get_coords_zeroth_term_forced():
            # used during training to correctly fit gamma_x_0
            coord_const_term = -0.5 * dim_coords * (log(2 * torch.pi) + gamma_0_coords.log())
            coord_fit_term = -0.5 * data.batch_sum @ ((eps_coords - pred_eps_coords) ** 2).sum(dim=-1) * sig_sq_0 / alf_sq_0 / gamma_0_coords
            return coord_const_term + coord_fit_term
        
        def get_kl_t_greater_than_zero():
            var_term_coords = 0.5 * dim_coords * (
                              gamma_x_batch.log()
                              - bet_rev_batch.log() 
                              + bet_rev_batch / gamma_x_batch
                              - 1.)
            fit_term_coords = 0.5 * data.batch_sum @ ((eps_coords - pred_eps_coords) ** 2).sum(dim=-1) \
                              * (alf_sq_L_batch / alf_sq_batch) \
                              * (bet_batch ** 2) / (sig_sq_batch * gamma_x_batch)
            kl_coords = var_term_coords + fit_term_coords
            
            if split == "train":
                loss_coords_0 = get_coords_zeroth_term_forced()
                kl_coords = t_is_zero_float * loss_coords_0 + (1 - t_is_zero_float) * kl_coords
            
            # features
            num_feats = self.config.num_atom_types + 1
            var_term_feats = 0.5 * data.num_atoms * (
                             gamma_h_batch.log().sum(dim=-1)
                             - num_feats * bet_rev_batch.log()
                             + (bet_rev_batch[:, None] / gamma_h_batch).sum(dim=-1)
                             - num_feats
            )
            fit_term_feats = 0.5 * data.batch_sum @ ((eps_features - pred_eps_features) ** 2 / gamma_h_nodes).sum(dim=-1) \
                             * (alf_sq_L_batch / alf_sq_batch) \
                             * (bet_batch ** 2) / sig_sq_batch
            kl_feats = var_term_feats + fit_term_feats
            
            kl = kl_coords + kl_feats
            return kl
    
        def get_vlb_zeroth_term():
            assert not self.training  # we should not perform a gradient step wrt. the output from this function
            # first sample z0
            z0_coords = alf_0 * s_coords + sig_0 * eps_coords
            z0_features = alf_0 * s_features + sig_0 * eps_features
            
            # for continuous coords, we go through another pass of the egnn
            pred0_eps_coords, _ = self.egnn(n_nodes=data.num_atoms, coords=z0_coords, features=z0_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=0.)
            
            coord_const_term = -0.5 * dim_coords * (log(2 * torch.pi) + gamma_0_coords.log())
            coord_fit_term = -0.5 * data.batch_sum @ ((eps_coords - pred0_eps_coords) ** 2).sum(dim=-1) * sig_sq_0 / alf_sq_0 / gamma_0_coords
            log_px_given_z0 = coord_const_term + coord_fit_term
            
            # for integer one_hot and charges, scale back to integer scaling
            z0_one_hot = z0_features[:, :-1]
            z0_charges = z0_features[:, -1]
            us_z0_one_hot, us_z0_charges = unscale_features(z0_one_hot, z0_charges)
            us_rev_sig0_one_hot, us_rev_sig0_charges = unscale_features(gamma_0_feats[:-1], gamma_0_feats[-1])  # note that we use our LEARNED variances here! HOWEVER, we should never perform a gradient step wrt. log_ph_charges_given_z0 and log_ph_one_hot_given_z0 since this will encourage our model to learn a variance var0 tending towards 0. instead, we only use this function to calculate the vlb AFTER training
            
            # for integer-value charges, find gaussian integral of radius 1 around the deviation
            us_charge_err_centered = data.charges - us_z0_charges[:, None]
            
            log_ph_charges_given_z0 = data.batch_sum @ torch.log(
                cdf_standard_gaussian((us_charge_err_centered + 0.5) / us_rev_sig0_charges)
                - cdf_standard_gaussian((us_charge_err_centered - 0.5) / us_rev_sig0_charges)
                + 1e-10
            ).sum(dim=-1)
            
            # for one-hot values, find gaussian integral around 1, since one-hot encoded
            us_one_hot_err_centered = us_z0_one_hot - 1.
            log_ph_one_hot_given_z0_unnormalised = torch.log(
                cdf_standard_gaussian((us_one_hot_err_centered + 0.5) / us_rev_sig0_one_hot[None, :])
                - cdf_standard_gaussian((us_one_hot_err_centered - 0.5) / us_rev_sig0_one_hot[None, :])
                + 1e-10
            )
            log_one_hot_normalisation_factor = torch.logsumexp(log_ph_one_hot_given_z0_unnormalised, dim=-1, keepdim=True)
            log_one_hot_probabilities = log_ph_one_hot_given_z0_unnormalised - log_one_hot_normalisation_factor
            log_ph_one_hot_given_z0 = data.batch_sum @ (log_one_hot_probabilities * data.one_hot).sum(dim=-1)
            
            return log_px_given_z0 + log_ph_one_hot_given_z0 + log_ph_charges_given_z0 
        
        ### TOOLS TO CALCULATE VANILLA VLB
        def get_van_kl_t_greater_than_zero():
            sq_err_coords = (data.batch_sum @ ((eps_coords - pred_eps_coords) ** 2)).sum(dim=-1)
            sq_err_features = (data.batch_sum @ ((eps_features - pred_eps_features) ** 2)).sum(dim=-1)
            weight = 0.5 * ((alf_lag_batch / sig_lag_batch) / (alf_batch / sig_batch) - 1)
            error = weight * (sq_err_coords + sq_err_features)
            return error
        
        def get_van_vlb_zeroth_term():
            # for continuous coords, we resample 
            z0_coords = alf_0 * s_coords + sig_0 * eps_coords
            z0_features = alf_0 * s_features + sig_0 * eps_features
            pred0_eps_coords, pred0_eps_features = self.egnn(n_nodes=data.num_atoms, coords=z0_coords, features=z0_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=0.)
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
        
        if split == "train":
            kl_t_greater_than_zero = get_kl_t_greater_than_zero()
            train_loss = 2. * (kl_t_greater_than_zero / data.num_atoms).mean()
            if train_loss.isnan():
                warn("Encountered NAN loss.")
                train_loss = torch.full_like(train_loss, fill_value=torch.nan, requires_grad=True)
            return train_loss
        else:
            eps = torch.concat([eps_coords, eps_features], dim=-1)
            pred_eps = torch.concat([pred_eps_coords, pred_eps_features], dim=-1)
            avr_sq_dist = ((eps - pred_eps) ** 2).mean()
            
            kl_t_greater_than_zero = get_kl_t_greater_than_zero()
            vlb_zero = -get_vlb_zeroth_term()
            
            vlb_est = -data.size_log_probs + vlb_zero + self.config.num_steps * kl_t_greater_than_zero
            
            return vlb_est.mean(), avr_sq_dist