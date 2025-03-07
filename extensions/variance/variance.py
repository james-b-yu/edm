import torch
from torch import nn
from model import EGNN, EGNNConfig
from utils.diffusion import cosine_noise_schedule, cosine_beta_schedule

class VarianceDiffusion(nn.Module):
    def __init__(self, egnn_config: EGNNConfig, num_steps: int, device: torch.device|str):
        super().__init__()
        
        self.egnn = EGNN(egnn_config)
        self.num_steps = num_steps
        self.schedule = cosine_beta_schedule(num_steps, device)
        
        self.V_h = nn.Parameter(0.25 + 0.5 * torch.rand(size=(self.egnn.config.features_d, )))
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