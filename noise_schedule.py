import torch

# Creates a default noise schedule if not specified
def default_noise_schedule(num_steps, device):
    t = torch.linspace(0, 1, num_steps, device=device)  # Move to the correct device
    alpha_t = torch.sqrt(1 - t**2).to(device)  # Ensure alpha is on the same device
    sigma_t = torch.sqrt(t**2).to(device)  # Ensure sigma is on the same device
    return {"alpha": alpha_t, "sigma": sigma_t}

# Creates a numerically stable cosine noise schedule
def cosine_noise_schedule(num_steps, device):
    t = torch.linspace(0, 1, num_steps, device=device)  # Time steps
    alpha_t = torch.cos((t + 0.008) / 1.008 * (torch.pi / 2)) ** 2

    # Prevent alpha_t from becoming too small
    alpha_t = torch.clamp(alpha_t, min=1e-2)  # Ensuring minimum value

    # Ensure stability in sigma calculation
    sigma_t = torch.sqrt(torch.clamp(1 - alpha_t**2, min=1e-7))  

    return {"alpha": alpha_t.to(device), "sigma": sigma_t.to(device)}
