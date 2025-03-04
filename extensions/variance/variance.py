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
        
        self.to(device=device)
        
    def forward(self):
        raise NotImplementedError("Please use self.egnn.forward")