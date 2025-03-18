
from math import log
from typing import Literal
from warnings import warn
from tqdm import tqdm

import torch
from data import EDMDataloaderItem,get_util_tensors
from configs.model_config import EDMConfig
from models.base import EGNN, BaseEDM
from utils.diffusion import cdf_standard_gaussian


class EDM(BaseEDM):
    """Vanilla EDM (no extensions)
    """        
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

        train_loss = avr_sq_dist

        if train_loss.isnan():
            warn("Encountered NAN loss.")
            train_loss = torch.full_like(train_loss, fill_value=torch.nan, requires_grad=True)
        return train_loss
        
    def estimate_vlb(self, data: EDMDataloaderItem, force_t: None|int = None):
        s_coords, s_one_hot, s_charges = self.scale_inputs(data.coords, data.one_hot, data.charges)
        s_features = torch.cat([s_one_hot, s_charges], dim=-1)
        
        lowest_t = 1
        
        t_batch_int = torch.randint(low=lowest_t, high=self.config.num_steps + 1, size=(data.num_atoms.shape[0], ), dtype=torch.long, device=self.config.device)  # [B]
        if force_t is not None:
            t_batch_int.fill_(force_t)
            
        t_nodes_int = t_batch_int[data.expand_idx] # [N]
        
        t = t_nodes_int / self.config.num_steps
        
        alf_0 = self.schedule.alpha[0]
        sig_0 = self.schedule.sigma[0]
        
        alf_nodes = self.schedule.alpha[t_nodes_int]
        sig_nodes = self.schedule.sigma[t_nodes_int]
        

        alf_sq_batch = self.schedule.alpha_squared[t_batch_int]
        alf_sq_L_batch = self.schedule.alpha_squared_L[t_batch_int]
        sig_sq_batch = self.schedule.sigma_squared[t_batch_int]
        sig_sq_L_batch = self.schedule.sigma_squared_L[t_batch_int]
        
        eps_coords = data.demean @ torch.randn_like(s_coords)
        eps_features = torch.randn_like(s_features)
        
        z_coords = alf_nodes[:, None] * s_coords + sig_nodes[:, None] * eps_coords
        z_features = alf_nodes[:, None] * s_features + sig_nodes[:, None] * eps_features
        
        pred_eps_coords, pred_eps_features = self.egnn(n_nodes=data.num_atoms, coords=z_coords, features=z_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=t)
        

        def get_van_kl_t_greater_than_zero():
            sq_err_coords = (data.batch_sum @ ((eps_coords - pred_eps_coords) ** 2)).sum(dim=-1)
            sq_err_features = (data.batch_sum @ ((eps_features - pred_eps_features) ** 2)).sum(dim=-1)
            weight = 0.5 * ((alf_sq_L_batch / sig_sq_L_batch) / (alf_sq_batch / sig_sq_batch) - 1)
            error = weight * (sq_err_coords + sq_err_features)
            return error
        
        def get_van_vlb_zeroth_term():
            # constant coord term
            const0 = -(data.num_atoms - 1.) * 3 * (0.5 * log(2 * torch.pi) + torch.log(sig_0) - torch.log(alf_0)) # = log Z in the paper
            
            # for continuous coords, we resample 
            z0_coords = alf_0 * s_coords + sig_0 * eps_coords
            z0_features = alf_0 * s_features + sig_0 * eps_features
            pred0_eps_coords, pred0_eps_features = self.egnn(n_nodes=data.num_atoms, coords=z0_coords, features=z0_features, edges=data.edges, reduce=data.reduce, demean=data.demean, time_frac=0.)
            error0 = -0.5 * (data.batch_sum @ ((eps_coords - pred0_eps_coords) ** 2)).sum(dim=-1)
            log_px_given_z0 = error0  # during inference we take sums
            
            # for integer one_hot and charges, scale back to integer scaling
            z0_one_hot = z0_features[:, :-1]
            z0_charges = z0_features[:, -1]
            _, us_z0_one_hot, us_z0_charges = self.unscale_inputs(0, z0_one_hot, z0_charges)
            _, us_sig0_one_hot, us_sig0_charges = self.unscale_inputs(0, sig_0, sig_0)
            
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
            
            return const0 + log_px_given_z0 + log_ph_one_hot_given_z0 + log_ph_charges_given_z0
        
        eps = torch.concat([eps_coords, eps_features], dim=-1)
        pred_eps = torch.concat([pred_eps_coords, pred_eps_features], dim=-1)
        avr_sq_dist = ((eps - pred_eps) ** 2).mean()
        

        van_kl_t_greater_than_zero = get_van_kl_t_greater_than_zero()
        van_vlb_zero = -get_van_vlb_zeroth_term()
        van_vlb_est = -data.size_log_probs + van_vlb_zero + self.config.num_steps * van_kl_t_greater_than_zero
        return van_vlb_est.mean(), avr_sq_dist
    
    def std_x(self, time: int | torch.Tensor):
        """Replace learned standard deviation for coordinates with noise schedule standard deviation."""
        return self.schedule.sigma[time]  # sqrt(sigma_squared)

    def std_h(self, time: int | torch.Tensor):
        """Replace learned standard deviation for features with noise schedule standard deviation."""
        return self.schedule.sigma[time]  # Apply to all feature dimensions

    def gamma_x(self, time: int | torch.Tensor):
        """Replace learned variance for coordinates with noise schedule variance."""
        return self.schedule.sigma_squared[time]

    def gamma_h(self, time: int | torch.Tensor):
        """Replace learned variance for features with noise schedule variance."""
        return self.schedule.sigma_squared[time]  # Apply to all feature dimensions

    @torch.no_grad()
    def _sample_flattened(self, num_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert num_atoms.dtype == torch.long and num_atoms.dim() == 1, "You must provide a tensor of length [B] and type long, where B is the number of molecules to create"
        assert not self.training, "We don't want to be training here"
        
        B = int(num_atoms.size(0))  # number of molecules to mkae 
        N = int(num_atoms.sum())   # total number of atoms
        
        edges, reduce, demean, expand_idx, batch_mean, batch_sum = get_util_tensors(num_atoms)
        
        coords = demean @ torch.randn(size=(N, 3), dtype=torch.float32, device=self.config.device)
        feats = torch.randn(size=(N, self.config.num_atom_types + 1), dtype=torch.float32, device=self.config.device)
        
        T = self.config.num_steps
        for t_int in tqdm(range(T, 0, -1), leave=False, unit="step"):
            # sample z(t-1) | z(t)
            t_frac = t_int/T
            alf_t = self.schedule.alpha[t_int]
            alf_s = self.schedule.alpha_L[t_int]
            bet_t = self.schedule.beta[t_int]
            sig_t = self.schedule.sigma[t_int]
            
            new_eps_coords = demean @ torch.randn_like(coords)
            new_eps_feats = torch.randn_like(feats)
            pred_eps_coords, pred_eps_feats = self.egnn(n_nodes=num_atoms, coords=coords, features=feats, edges=edges, reduce=reduce, demean=demean, time_frac=t_frac)
            
            std_coords = self.std_x(t_int)
            std_feats = self.std_h(t_int)
            
            coords = (alf_s / alf_t) * coords - (alf_s / alf_t) * (bet_t / sig_t) * pred_eps_coords + std_coords * new_eps_coords
            feats  = (alf_s / alf_t) * feats  - (alf_s / alf_t) * (bet_t / sig_t) * pred_eps_feats  + std_feats  * new_eps_feats
            
        # now z(0) = (coords, feats) so we need to sample x | z(0)
        alf_0 = self.schedule.alpha[0]
        sig_0 = self.schedule.sigma[0]
        
        new_eps_coords = demean @ torch.randn_like(coords)
        new_eps_feats = torch.randn_like(feats)
        pred_eps_coords, pred_eps_feats = self.egnn(n_nodes=num_atoms, coords=coords, features=feats, edges=edges, reduce=reduce, demean=demean, time_frac=0.)
        
        std_coords = self.std_x(0)
        std_feats = self.std_h(0)
        
        # sample final variables
        coords = coords / alf_0 - (sig_0 / alf_0) * pred_eps_coords + std_coords * new_eps_coords
        feats  = feats  / alf_0 - (sig_0 / alf_0) * pred_eps_feats  + std_feats  * new_eps_feats
        
        one_hot, charges = feats[:, :-1], feats[:, -1]
        coords, one_hot, charges = self.unscale_inputs(coords, one_hot, charges)
        one_hot, charges = one_hot.round().to(dtype=torch.long), charges.round().to(dtype=torch.long)

        print("coords: ")
        print(coords)
        print("one_hot: ")
        print(one_hot)
        
        return coords, one_hot, charges