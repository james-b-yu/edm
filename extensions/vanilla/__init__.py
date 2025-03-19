from argparse import Namespace

import wandb.wandb_run

from torch.utils.data import DataLoader
from wandb.wandb_run import Run

from .full_eval import run_eval

def run(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    if args.pipeline == "train":
        raise NotImplementedError
    elif args.pipeline == "valid":
        raise NotImplementedError
    elif args.pipeline == "sample":
        raise NotImplementedError
    elif args.pipeline == "demo":
        do_demo(args, dataloaders["valid"])
    elif args.pipeline == 'test':
        run_eval(args, dataloaders["test"])