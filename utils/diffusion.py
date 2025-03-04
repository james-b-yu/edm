import torch

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
    
    
def cosine_beta_schedule(timesteps, device: torch.device|str, s=0.008, raise_to_power: float = 1):
    """
    ADAPTED FROM https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L55
    
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = torch.linspace(0, steps, steps, device=device)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, min=0, max=0.999)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    if raise_to_power != 1:
        alphas_cumprod = alphas_cumprod ** raise_to_power

    return {
        "alpha": alphas_cumprod ** 0.5,
        "sigma": (1 - alphas_cumprod) ** 0.5,
    }