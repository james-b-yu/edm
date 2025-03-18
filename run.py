#!/usr/bin/env python
"""
This script is used to perform all running
"""

import warnings

import torch
from data import get_qm9_dataloader
from configs.dataset_config import DATASET_INFO
from loops import enter_train_loop, enter_valid_loop
from configs.model_config import get_config_from_args
from models.variance_edm import VarianceEDM
from models.base import BaseEDM
from models.edm import EDM

from os import path
import pickle
import wandb
# from eval import compute_nll, get_test_dataloader

from args import args, parser

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    
    if args.checkpoint is not None:
        try:
            with open(path.join(args.checkpoint, "args.pkl"), "rb") as f:
                args_disk = pickle.load(f)
                args = parser.parse_args(namespace=args_disk)
        except Exception as _:
            warnings.warn("Did not restore args.pkl from checkpoint")
    if args.dataset in ["qm9", "qm9_no_h"]:
        dataloaders = {
            split: get_qm9_dataloader(use_h=args.dataset=="qm9", split=split, batch_size=args.batch_size, pin_memory=False, num_workers=args.dl_num_workers, prefetch_factor=args.dl_prefetch_factor) for split in ("train", "valid", "test")
        }
    else:
        raise NotImplementedError()
    
    Model = BaseEDM
    if args.extension == "vanilla":
        Model = EDM
    elif args.extension == "variance":
        Model = VarianceEDM
    else:
        raise NotImplementedError
    
    assert hasattr(dataloaders["train"].dataset, "num_atom_types")
    model = Model(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type: ignore
    
    if args.checkpoint is not None:                    
        print(f"Loading model checkpoint located in {args.checkpoint}")
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))  # always load the model as checkpoint
        
    if args.pipeline == "train":
        model_ema = Model(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type: ignore
        model_ema.load_state_dict(model.state_dict())  # initialise the ema model to be the same as the model
        
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr, amsgrad=True,
            weight_decay=1e-12
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, threshold=args.scheduler_threshold, threshold_mode="rel", min_lr=args.scheduler_min_lr)
        
        if args.checkpoint is not None:
            # load elements specific to training (model_ema, optim, scheduler)
            try:
                model_ema.load_state_dict(torch.load(path.join(args.checkpoint, "model_ema.pth"), map_location=args.device))
            except Exception as e:
                warnings.warn(f"Could not load model exponential moving average state dict. Initialising the EMA model with same weights as model.pth. Error was '{str(e)}'")
            
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
        
        wandb_run = None
        if args.use_wandb:
            wandb.login()
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
                id=args.run_id if args.run_id != "None" else None,
                resume="allow",
            )
        
        enter_train_loop(model, model_ema, optim, scheduler, args, dataloaders["train"], dataloaders["valid"], wandb_run)
    elif args.pipeline == "valid" or args.pipeline == "test":
        model.eval()
        split = args.pipeline

        print(f"Performing estimation of the VLB for 'model.pth' using {args.reruns} epoch(s) of the {split} set...")
        mean, std = enter_valid_loop(model, split, args, dataloaders[split])
        
        print(f"point estimate (std) for 'model.pth':     vlb: {mean:.2f} ({std:.2f})")
    elif args.pipeline == "sample":
        model.eval()

        num_molecules = args.num_samples
        batch_size = args.batch_size
        mol_sizes = torch.tensor(list(DATASET_INFO[args.dataset]["molecule_size_histogram"].keys()), dtype=torch.long, device=args.device)
        mol_size_probs = torch.tensor(list(DATASET_INFO[args.dataset]["molecule_size_histogram"].values()), dtype=torch.float, device=args.device)
        print(f"Sampling from 'model.pth'")
        samples = model.sample(num_molecules, batch_size, mol_sizes, mol_size_probs)
        
        # TODO: write these samples to disk as .xyz files
        # TODO: calculate stability metrics on the generated samples
        pass
    else:
        raise NotImplementedError
else:
    raise RuntimeError