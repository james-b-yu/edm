from argparse import Namespace

import wandb.wandb_run
from .train import enter_train_valid_test_loop, enter_sample
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

def run(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    if args.pipeline == "train":
        enter_train_valid_test_loop(args, dataloaders, wandb_run)
    elif args.pipeline == "valid":
        enter_train_valid_test_loop(args, dataloaders, wandb_run, no_train=True)
    elif args.pipeline == "sample":
        enter_sample(args, dataloaders, wandb_run)