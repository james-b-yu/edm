from argparse import Namespace
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class EGCLConfig:
    device: torch.device|str = "cuda"
    hidden_dim: int = 256
    tanh_multiplier: float = 15.
    dataset_name: str = "qm9"
    use_h: bool = True

@dataclass
class EGNNConfig(EGCLConfig):
    num_atom_types: int = 5
    num_layers: int = 9

@dataclass
class EDMConfig(EGNNConfig):
    coord_in_scale: float = 1.
    one_hot_in_scale: float = 0.25
    charge_in_scale: float = 0.1
    num_steps: int = 1000
    schedule_type: Literal["cosine", "polynomial"] = "polynomial"
    
def get_config_from_args(args: Namespace, num_atom_types: int):
    return EDMConfig(
        device=args.device,
        hidden_dim=args.hidden_d,
        tanh_multiplier=args.tanh_range,
        num_layers=args.num_layers,
        num_atom_types=num_atom_types,  # type:ignore
        num_steps=args.num_steps,
        schedule_type=args.noise_schedule,
        coord_in_scale=1.,
        one_hot_in_scale=0.25,
        charge_in_scale=0.1,
        dataset_name=args.dataset.split('_')[0],
        use_h=('_no_h' not in args.dataset))