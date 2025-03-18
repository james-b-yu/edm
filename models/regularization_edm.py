
from math import log
from warnings import warn
from tqdm import tqdm

import torch
import torch.nn.functional as F
from data import EDMDataloaderItem,get_util_tensors
from configs.model_config import EDMConfig
from models.base import BaseEDM
from utils.diffusion import cdf_standard_gaussian

from extensions.regularization.penalty import get_disconnected_penalty

class RegularizationEDM(BaseEDM):
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
        epsilon_x = torch.randn_like(in_coords)
        epsilon_x -= epsilon_x.mean(dim=0, keepdim=True)
        epsilon_h = torch.randn_like(in_features)
        
        # Apply reverse diffusion update
        coords = (1 / (alpha_t)) * (in_coords - sigma_t * pred_eps_coords) + sigma_t * epsilon_x
        features = (1 / (alpha_t)) * (in_features - sigma_t * pred_eps_features) + sigma_t * epsilon_h
        features = F.one_hot(features.argmax(dim=-1), num_classes=5).float()

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
        disc_penalty = get_disconnected_penalty(post_coords, post_features, self.config.dataset_name, self.config.use_h)

        train_loss = avr_sq_dist + disc_penalty

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
        """noise schedule standard deviation."""
        return self.schedule.sigma[time]  # sqrt(sigma_squared)

    def std_h(self, time: int | torch.Tensor):
        """noise schedule standard deviation."""
        return self.schedule.sigma[time]  # Apply to all feature dimensions

    def gamma_x(self, time: int | torch.Tensor):
        """noise schedule variance."""
        return self.schedule.sigma_squared[time]

    def gamma_h(self, time: int | torch.Tensor):
        """noise schedule variance."""
        return self.schedule.sigma_squared[time]  # Apply to all feature dimensions

    @torch.no_grad()
    def _sample_flattened(self, num_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements the standard EDM sampling algorithm from the paper.

        Args:
            num_atoms (torch.Tensor): Tensor containing the number of atoms per molecule in the batch.

        Returns:
            Tuple of tensors (coords, one_hot, charges) representing the final molecule.
        """
        assert num_atoms.dtype == torch.long and num_atoms.dim() == 1, "Provide a tensor of shape [B] with dtype long."
        assert not self.training, "Sampling should not be performed during training."

        B = int(num_atoms.size(0))  # Number of molecules
        N = int(num_atoms.sum())  # Total number of atoms

        edges, reduce, demean, expand_idx, batch_mean, batch_sum = get_util_tensors(num_atoms)

        # Step 1: Initialize z_T from a standard normal distribution
        z_coords = demean @ torch.randn(size=(N, 3), dtype=torch.float32, device=self.config.device)
        z_features = torch.randn(size=(N, self.config.num_atom_types + 1), dtype=torch.float32, device=self.config.device)

        T = self.config.num_steps
        for t_int in tqdm(range(T, 0, -1), leave=False, unit="step"):
            t_frac = t_int / T  # Normalize time step

            # Step 2: Retrieve noise schedule parameters
            alf_t = self.schedule.alpha[t_int]
            alf_s = self.schedule.alpha_L[t_int]  # Alpha at t-1
            sig_t = self.schedule.sigma[t_int]  # Standard deviation
            sig_t_sqr = self.schedule.sigma_squared[t_int]
            sig_t_s = self.schedule.sigma[t_int - 1]  # Sigma at t-1
            sig_t_s_sqr = self.schedule.sigma_squared[t_int - 1]
            bet_t = self.schedule.beta[t_int]  # Beta at step t

            # Step 3: Sample Gaussian noise ε_t
            eps_coords = demean @ torch.randn_like(z_coords)
            eps_features = torch.randn_like(z_features)

            # Step 4: Predict noise ε̂ using EGNN
            pred_eps_coords, pred_eps_features = self.egnn(
                n_nodes=num_atoms, coords=z_coords, features=z_features, edges=edges, reduce=reduce, demean=demean, time_frac=t_frac
            )

            # Step 5: Compute mean for the reverse step
            mu_t_s_coords = (1 / alf_t) * z_coords - (sig_t_sqr / (alf_t * sig_t)) * pred_eps_coords
            mu_t_s_feats = (1 / alf_t) * z_features - (sig_t_sqr / (alf_t * sig_t)) * pred_eps_features

            # Step 6: Apply reverse diffusion step
            z_coords = mu_t_s_coords + sig_t_s * eps_coords
            z_features = mu_t_s_feats + sig_t_s * eps_features

        # Final step: sample x | z(0)
        alf_0 = self.schedule.alpha[0]
        sig_0 = self.schedule.sigma[0]

        final_eps_coords = demean @ torch.randn_like(z_coords)
        final_eps_feats = torch.randn_like(z_features)

        pred_eps_coords, pred_eps_feats = self.egnn(
            n_nodes=num_atoms, coords=z_coords, features=z_features, edges=edges, reduce=reduce, demean=demean, time_frac=0.
        )

        # Final denoising step
        z_coords = (1 / alf_0) * z_coords - (sig_0 / alf_0) * pred_eps_coords + sig_0 * final_eps_coords
        z_features = (1 / alf_0) * z_features - (sig_0 / alf_0) * pred_eps_feats + sig_0 * final_eps_feats

        # Convert features back to original format
        one_hot, charges = z_features[:, :-1], z_features[:, -1]
        coords, one_hot, charges = self.unscale_inputs(z_coords, one_hot, charges)
        one_hot, charges = one_hot.round().to(dtype=torch.long), charges.round().to(dtype=torch.long)

        print("coord: ")
        print(coords)
        print("one_hot: ")
        print(one_hot)

        return coords, one_hot, charges