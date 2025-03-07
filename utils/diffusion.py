import torch
import numpy as np

def cosine_noise_schedule(num_steps: int, device: torch.device|str):
    t = torch.linspace(0, 1, num_steps, device=device)
    alpha_t = torch.cos((t + 0.008) / 1.008 * (torch.pi / 2)) ** 2
    sigma_t = torch.sqrt(1 - alpha_t**2)
    return {"alpha": alpha_t, "sigma": sigma_t}

def default_noise_schedule(num_steps: int, device: torch.device|str):
        t = torch.linspace(0, 1, num_steps, device=device)
        
        # schedule
        alpha_t = torch.sqrt(1 - t**2) 
        
        # Noise
        sigma_t = torch.sqrt(t**2) 
        
        return {"alpha": alpha_t, "sigma": sigma_t}
    
    
def cosine_beta_schedule(timesteps, device: torch.device|str, s=0.008):
    """
    ADAPTED FROM https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L55
    
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps, dtype=np.longdouble)
    helper = np.cos(((x / steps) + s) / (1. + s) * np.pi * 0.5) ** 2
    helper = helper / helper[0]
    
    betas_transition = 1. - (helper[1:] / helper[:-1])
    betas_transition = np.clip(betas_transition, a_min=0., a_max=0.999)
    alpha_squared_transition = 1. - betas_transition
    
    
    alpha_bar = np.cumprod(alpha_squared_transition, axis=0)
    betas_sigma_transition = (1. - alpha_bar[:-1]) / (1. - alpha_bar[1:]) * betas_transition[1:]
    
    return {
        "alpha": torch.from_numpy((alpha_bar ** 0.5).astype(np.float32)).to(device=device),
        "alpha_squared": torch.from_numpy((alpha_bar).astype(np.float32)).to(device=device),
        "alpha_L_squared": torch.from_numpy(np.concat([np.array([1], dtype=np.longdouble), alpha_bar[:-1]]).astype(np.float32)).to(device=device),  # \alpha_{t-1} with artificial 1 at t=0
        "sigma": torch.from_numpy(((1 - alpha_bar) ** 0.5).astype(np.float32)).to(device=device),
        "sigma_squared": torch.from_numpy(((1 - alpha_bar)).astype(np.float32)).to(device=device),
        "beta": torch.from_numpy(betas_transition.astype(np.float32)).to(device=device),   # transition variances
        "beta_squared": torch.from_numpy((betas_transition ** 2).astype(np.float32)).to(device=device),   # the SQUARE of the transition variances
        "beta_sigma": torch.from_numpy(np.concatenate([betas_transition[0, None], betas_sigma_transition], axis=0).astype(np.float32)).to(device=device),  # copy zeroth element of beta into beta_sigma
    }
    
def scale_inputs(coords: torch.Tensor, one_hot: torch.Tensor, charges: torch.Tensor):
    """use the scaling implmeneted by the paper

    Args:
        coords (torch.Tensor):
        one_hot (torch.Tensor):
        charges (torch.Tensor):
    """
    
    return coords, 0.25 * one_hot, 0.1 * charges

def unscale_outputs(coords: torch.Tensor, one_hot: torch.Tensor, charges: torch.Tensor):
    """undo the scaling implemented by the paper

    Args:
        coords (torch.Tensor):
        one_hot (torch.Tensor):
        charges (torch.Tensor):
    """
    
    return coords, 4.0 * one_hot, 10.0 * charges


#Gradient clipping
class Queue():
    """COPIED FROM HOOGEBOOM REPO"""
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow: torch.nn.Module, gradnorm_queue: Queue):
    """COPIED FROM HOOGEBOOM REPO

    Args:
        flow (torch.Tensor): _description_
        gradnorm_queue (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    # if float(grad_norm) > max_grad_norm:
    #     print(f'Clipped gradient with value {grad_norm:.1f} '
    #           f'while allowed {max_grad_norm:.1f}')
    return grad_norm