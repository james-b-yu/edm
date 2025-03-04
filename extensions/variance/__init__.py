from argparse import Namespace
from .train import enter_train_valid_test_loop
from torch.utils.data import DataLoader

def run(args: Namespace, dataloaders: dict[str, DataLoader]):
    if args.pipeline == "train":
        enter_train_valid_test_loop(args, dataloaders)
    elif args.pipeline == "evaluate":
        raise NotImplementedError()