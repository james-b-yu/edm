from argparse import Namespace
from math import ceil
from os import path
import warnings

import torch
from tqdm import tqdm
import wandb.wandb_run

from dataset_info import DATASET_INFO
from .train import enter_train_loop, enter_valid_loop, one_valid_epoch
from model_config import get_config_from_args
from .variance_model import VarianceEDM
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

def run(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    model = VarianceEDM(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type:ignore
    model_ema = VarianceEDM(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type:ignore
    model_ema.load_state_dict(model.state_dict())  # initialise the ema model to be the same as the model
    
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, threshold=args.scheduler_threshold, threshold_mode="rel", min_lr=args.scheduler_min_lr)
    
    if args.checkpoint is not None:
        print(f"Loading model checkpoint located in {args.checkpoint}")
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))  # always load the model as checkpoint
        ema_loaded = False
        try:
            ema_loaded = True
            model_ema.load_state_dict(torch.load(path.join(args.checkpoint, "model_ema.pth"), map_location=args.device))
        except Exception as e:
            warnings.warn(f"Could not load model exponential moving average state dict. Initialising the EMA model with same weights as model.pth. Error was '{str(e)}'")
            model_ema.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))
            
        if args.restore_optim_state:
            try:
                optim.load_state_dict(torch.load(path.join(args.checkpoint, "optim.pth"), map_location=args.device))
            except Exception as e:
                warnings.warn(f"Could not load optim state dict. Using empty initialisation. Error was '{str(e)}'")
        if args.restore_scheduler_state:
            try:
                scheduler.load_state_dict(torch.load(path.join(args.checkpoint, "scheduler.pth"), map_location=args.device))        
            except Exception as e:
                warnings.warn(f"Could not load scheduler state dict. Using empty initialisation. Error was '{str(e)}'")
        if args.force_start_lr is not None:        
            for param_group in optim.param_groups:
                param_group['lr'] = args.force_start_lr
    
    if args.pipeline == "train":
        enter_train_loop(model, model_ema, optim, scheduler, args, dataloaders["train"], dataloaders["valid"], wandb_run)
    elif args.pipeline == "valid" or args.pipeline == "test":
        model.eval()
        model_ema.eval()
        split = args.pipeline

        print(f"Performing {args.reruns} (noisy) validation epoch(s) on 'model.pth'")
        mean, std = enter_valid_loop(model, split, args, dataloaders[split])
        if ema_loaded:
            print(f"Performing {args.reruns} (noisy) validation epoch(s) on 'model_ema.pth'")
            ema_mean, ema_std = enter_valid_loop(model_ema, split, args, dataloaders[split])
        
        print(f"mean (std) for 'model.pth':     vlb: {mean:.2f} ({std:.2f})")
        if ema_loaded:
            print(f"mean (std) for 'model_ema.pth': vlb: {ema_mean:.2f} ({ema_std:.2f})")
    elif args.pipeline == "sample":
        model.eval()
        model_ema.eval()

        atom_sizes = torch.tensor(list(DATASET_INFO[args.dataset]["molecule_size_histogram"].keys()), dtype=torch.long, device=args.device)
        atom_probs = torch.tensor(list(DATASET_INFO[args.dataset]["molecule_size_histogram"].values()), dtype=torch.float, device=args.device)
        num_samples = args.num_samples
        samples = []
        for i in range(ceil(num_samples / args.batch_size)):
            B = min(args.batch_size, num_samples - len(samples))
            num_atoms = atom_sizes[torch.multinomial(atom_probs, B, replacement=True)]
            coords, one_hot, charges = model.sample_flattened(num_atoms)
            
            # separate the molecules and append to samples
            batch_idx = 0
            flattened_idx = 0
            while batch_idx < B:
                size = num_atoms[batch_idx]
                samples.append((coords[flattened_idx:flattened_idx + size],
                                one_hot[flattened_idx:flattened_idx + size],
                                charges[flattened_idx:flattened_idx + size]))
                flattened_idx += size
                batch_idx += 1
            
            assert(batch_idx == B and flattened_idx == num_atoms.sum())
        
        assert(len(samples) == num_samples)
        
        return samples
    else:
        raise NotImplementedError