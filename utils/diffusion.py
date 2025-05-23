from dataclasses import dataclass
from functools import cache
import torch
import numpy as np
from math import log

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

# def polynomial_noise_schedule(T=1000, s=1e-5):
#     """
#     Implements the polynomial noise schedule from the paper.
    
#     Parameters:
#         T (int): Total number of diffusion steps.
#         s (float): Small positive value to avoid numerical instabilities.
    
#     Returns:
#         alpha_t (np.array): Alpha values controlling the signal retention.
#         sigma_t (np.array): Sigma values controlling noise level.
#     """
#     t = np.arange(T + 1) / T  # Normalize time steps
#     f_t = 1 - t**2  # Quadratic polynomial function
#     alpha_t = (1 - 2 * s) * f_t + s  # Noise schedule
#     sigma_t = np.sqrt(1 - alpha_t**2)  # Compute sigma
    
#     return alpha_t, sigma_t
    
def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    COPIED FROM https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L24
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2    

@dataclass
class ScheduleInfo:
    alpha: torch.Tensor
    alpha_L: torch.Tensor  # alpha evaluated at t-1 -- first value is padded = 1
    alpha_squared: torch.Tensor
    alpha_squared_L: torch.Tensor  # alpha_squared evaluated at t-1 -- first value is padded = 1
    sigma: torch.Tensor
    sigma_squared: torch.Tensor
    sigma_L: torch.Tensor
    sigma_squared_L: torch.Tensor
    beta: torch.Tensor  # var(t | t-1)
    rev_beta: torch.Tensor  # var(t-1 | t)
    
    def __getitem__(self, key):
        # Check if the key exists in the dataclass' __dict__
        if key not in self.__dict__:
            raise KeyError(f"'{key}' is not a valid field")
        
        # If the key exists, return the corresponding value
        return self.__dict__[key]

def _get_schedule_from_alpha_squared_np(alpha_squared_np: np.ndarray, device: torch.device|str):
    alpha_squared_np_L = np.concat([np.array([1], dtype=np.longdouble), alpha_squared_np[:-1]])
    sigma_squared_np = 1. - alpha_squared_np
    sigma_squared_np_L = 1. - alpha_squared_np_L
    alpha_squared_ratio_np = alpha_squared_np / alpha_squared_np_L
    beta_np = sigma_squared_np - alpha_squared_ratio_np * sigma_squared_np_L  # variance of t given t - 1
    rev_beta_np = sigma_squared_np_L - alpha_squared_ratio_np * sigma_squared_np_L ** 2 / sigma_squared_np
    rev_beta_np[0] = 0.5 * beta_np[0]  # fill dummy value for zeroth term but still allow for variation
    
    alpha = torch.from_numpy((alpha_squared_np ** 0.5).astype(np.float32)).to(device=device)
    alpha_L = torch.from_numpy((alpha_squared_np_L ** 0.5).astype(np.float32)).to(device=device)
    alpha_squared = torch.from_numpy(alpha_squared_np.astype(np.float32)).to(device=device)
    alpha_squared_L = torch.from_numpy(alpha_squared_np_L.astype(np.float32)).to(device=device)
    sigma = torch.from_numpy((sigma_squared_np ** 0.5).astype(np.float32)).to(device=device)
    sigma_squared = torch.from_numpy(sigma_squared_np.astype(np.float32)).to(device=device)
    sigma_L = torch.from_numpy((sigma_squared_np_L ** 0.5).astype(np.float32)).to(device=device)
    sigma_squared_L = torch.from_numpy(sigma_squared_np_L.astype(np.float32)).to(device=device)
    beta = torch.from_numpy(beta_np.astype(np.float32)).to(device=device)
    rev_beta = torch.from_numpy(rev_beta_np.astype(np.float32)).to(device=device)
    
    return ScheduleInfo(
        alpha=alpha,
        alpha_L=alpha_L,
        alpha_squared=alpha_squared,
        alpha_squared_L=alpha_squared_L,
        sigma=sigma,
        sigma_squared=sigma_squared,
        sigma_L=sigma_L,
        sigma_squared_L=sigma_squared_L,
        beta=beta,
        rev_beta=rev_beta,
    )

def polynomial_schedule(timesteps: int, device: torch.device|str, power=2., s=1e-5):
    """
    ADAPTED FROM https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py#L39
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alpha_squared_np = (1 - np.power(x / steps, power))**2

    alpha_squared_np = clip_noise_schedule(alpha_squared_np, clip_value=0.001)

    precision = 1 - 2 * s

    alpha_squared_np = precision * alpha_squared_np + s
    
    return _get_schedule_from_alpha_squared_np(alpha_squared_np, device)
    
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
    
    return _get_schedule_from_alpha_squared_np(alpha_bar, device)

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


def gradient_clipping(flow: torch.nn.Module, gradnorm_queue: Queue, max: float):
    """COPIED FROM HOOGEBOOM REPO

    Args:
        flow (torch.Tensor): _description_
        gradnorm_queue (_type_): _description_
        max (float): _description_

    Returns:
        _type_: _description_
    """
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    max_grad_norm = max if max <= max_grad_norm else max_grad_norm

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=float(max_grad_norm), norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    # if float(grad_norm) > max_grad_norm:
    #     print(f'Clipped gradient with value {grad_norm:.1f} '
    #           f'while allowed {max_grad_norm:.1f}')
    return grad_norm

def gaussian_KL(q_mu: torch.Tensor, q_var: torch.Tensor, p_mu: torch.Tensor, p_var: torch.Tensor):
    """calculates KL divergence between coordinate-wise univariate Gaussians

    Args:
        q_mu (torch.Tensor): _description_
        q_var (torch.Tensor): _description_
        p_mu (torch.Tensor): _description_
        p_var (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    return 0.5 * (p_var.log() - q_var.log() + ((q_var + (q_mu - p_mu) ** 2) / p_var) - 1)

def gaussian_KL_batch(q_mu: torch.Tensor, q_var: torch.Tensor, p_mu: torch.Tensor, p_var: torch.Tensor, n_nodes: torch.Tensor, batch_sum: torch.Tensor, subspace=False):
    """calculates KL divergence between isotropic multivariate Gaussians per batch

    Args:
        q_mu (torch.Tensor): [N, dims]
        q_var (torch.Tensor): [B]
        p_mu (torch.Tensor): [N, dims]
        p_var (torch.Tensor): [B]
        n_nodes (torch.Tensor): [B]
        batch_sum (torch.Tensor): [B, N]
        subspace (bool): whether we are doing KL in the batch-level zero-mean coordinate subspace. Defaults to False

    Returns:
        torch.Tensor: _description_
    """
    return 0.5 * (((n_nodes - (1. if subspace else 0.)) * q_mu.shape[1]) * (p_var.log() - q_var.log() + (q_var / p_var) - 1) + batch_sum @ ((q_mu - p_mu) ** 2).sum(dim=-1))

def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / 1.4142135624))

def demean_using_mask(x: torch.Tensor, node_mask: torch.Tensor):
    """x corresponds to coordinates. demean using node_mask

    Args:
        x (torch.Tensor): coordinates, noised coordinates, predicted coordinates, etc.
        node_mask (torch.Tensor): node_mask
    """
    mean = x.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True)
    return x - mean * node_mask

@cache
def get_batch_edge_idx(max_num_nodes: int, batch_size: int, device: str):
    """returns two lists a and b, for each i, there exists and edge between a[i] and b[i] in the batched fully connected graph

    Args:
        max_num_nodes (int): largest atom in the batch
        batch_size (int): batch size
    """
    rows = []
    cols = []
    
    for b in range(batch_size):
        for i in range(max_num_nodes):
            for j in range(max_num_nodes):
                rows.append(i + b * max_num_nodes)
                cols.append(j + b * max_num_nodes)
    
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    
    return rows, cols

def get_coord_distance(z_coord: torch.Tensor, edge_idx: tuple[torch.Tensor, torch.Tensor]):
    """given coords and edge indices, return distances between every coord connected in the graph

    Args:
        z_coord (torch.Tensor): _description_
        edge_idx (torch.Tensor): _description_
    """
    
    row, col = edge_idx
    diff = z_coord[row] - z_coord[col]
    squared_distance = (diff ** 2).sum(dim=-1, keepdim=True)
    normed_difference = diff / (1 + (squared_distance + 1e-8).sqrt())
    
    return squared_distance, normed_difference

def unsorted_segment_sum(data: torch.Tensor, segment_idx: torch.Tensor, num_segments: int):
    res = data.new_zeros(size=(num_segments, data.shape[1]))
    segment_idx = segment_idx[:, None].expand(-1, data.shape[1])
    res.scatter_add_(0, segment_idx, data)
    return res