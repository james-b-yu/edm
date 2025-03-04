#!/usr/bin/env python
"""
This script is used to perform all running
"""

import sys

from data import get_qm9_dataloader
sys.path.append(".")

from args import args
from extensions.variance import run

if __name__ == "__main__":
    if args.checkpoint is not None:
        raise NotImplementedError()  # TODO: load args!
    
    if args.dataset in ["qm9", "qm9_no_h"]:
        dataloaders = {
            split: get_qm9_dataloader(use_h=args.dataset=="qm9", split=split, batch_size=args.batch_size) for split in ("train", "valid", "test")
        }
    else:
        raise NotImplementedError() 

    
    if args.extension == "variance":
        run(args, dataloaders)