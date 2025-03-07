from argparse import Namespace

import wandb.wandb_run
from .train import enter_train_valid_test_loop
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

def run(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    if args.pipeline == "train":
        enter_train_valid_test_loop(args, dataloaders, wandb_run)
    elif args.pipeline == "evaluate":
        raise NotImplementedError()