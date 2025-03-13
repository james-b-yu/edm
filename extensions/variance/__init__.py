from argparse import Namespace
from os import path

import torch
import wandb.wandb_run
from .train import enter_train_loop
from model_config import get_config_from_args
from .variance_model import VarianceEDM
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

def run(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    model = VarianceEDM(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type:ignore
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, threshold=args.scheduler_threshold, threshold_mode="rel", min_lr=args.scheduler_min_lr)
    
    if args.checkpoint is not None:
        print(f"Loading model checkpoints located in {args.checkpoint}")

        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))
        if args.restore_optim_state:
            optim.load_state_dict(torch.load(path.join(args.checkpoint, "optim.pth"), map_location=args.device))
        if args.restore_scheduler_state:
            scheduler.load_state_dict(torch.load(path.join(args.checkpoint, "scheduler.pth"), map_location=args.device))        
        if args.force_start_lr is not None:        
            for param_group in optim.param_groups:
                param_group['lr'] = args.force_start_lr
    
    if args.pipeline == "train":
        enter_train_loop(model, optim, scheduler, args, dataloaders["train"], dataloaders["valid"], wandb_run)
    elif args.pipeline == "valid":
        raise NotImplementedError
    elif args.pipeline == "sample":
        raise NotImplementedError
    else:
        raise NotImplementedError