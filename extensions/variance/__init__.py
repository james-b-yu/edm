from argparse import Namespace
from os import path
import warnings

import torch
import wandb.wandb_run
from .train import enter_train_loop, one_valid_epoch
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
        try:
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
    elif args.pipeline == "valid":
        one_valid_epoch(args, "valid", 0, dataloaders["valid"], model)
        one_valid_epoch(args, "valid", 0, dataloaders["valid"], model_ema)
    elif args.pipeline == "sample":
        raise NotImplementedError
    else:
        raise NotImplementedError