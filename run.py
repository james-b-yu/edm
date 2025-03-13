#!/usr/bin/env python
"""
This script is used to perform all running
"""

import sys
import warnings
from data import get_qm9_dataloader, get_masked_qm9_dataloader
sys.path.append(".")
from os import path
import pickle
import wandb
from eval import compute_nll, load_model, load_test_data

from args import args, parser
from extensions import vanilla, variance

if __name__ == "__main__":
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
        masked_dataloders = {
            split: get_masked_qm9_dataloader(use_h=args.dataset=="qm9", split=split, batch_size=args.batch_size, pin_memory=False, num_workers=args.dl_num_workers, prefetch_factor=args.dl_prefetch_factor) for split in ("train", "valid", "test")
        }
    else:
        raise NotImplementedError() 

    wandb_run = None
    if args.use_wandb:
        wandb.login()
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)  
        )
    
    if args.extension == "vanilla":
        vanilla.run(args, dataloaders if args.use_non_masked else masked_dataloders , wandb_run)
    if args.extension == "variance":
        if not args.use_non_masked:
            warnings.warn("The variance extension is implemented only with our implementation for the non-masked (flattened) data layout. --use-non-masked will be automatically enabled")
        variance.run(args, dataloaders, wandb_run)
