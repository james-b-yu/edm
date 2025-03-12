import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path

from masked_model import MaskedEDM, get_config_from_args

from utils.diffusion import cosine_noise_schedule,polynomial_schedule



def training_loop(args: Namespace, dl: DataLoader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    config = get_config_from_args(args, dl.dataset.num_atom_types)

    model = MaskedEDM(config)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",  # Typically "min" for reducing LR on loss plateau, or "max" for accuracy
        factor=args.scheduler_factor,  # Factor by which the LR is reduced (e.g., 0.5 means LR is halved)
        patience=args.scheduler_patience,  # Number of epochs with no improvement before reducing LR
        threshold=args.scheduler_threshold,  # Minimum relative improvement to reset patience
        min_lr=args.scheduler_min_lr,  # Minimum possible LR
    )

    if args.noise_schedule == "polynomial":
        polynomial_schedule(args.num_steps, device)
    elif args.noise_schedule == "cosine":
        cosine_noise_schedule(args.num_steps, device)
    else:
        raise(ValueError)

    print("[INFO] Noise schedule created.")

    # check if this is right
    print(f"[INFO] Dataset Loaded: {len(dl)} molecules in training set.")

    model.train()

    for epoch in range(args.start_epoch,args.end_epoch):

        
        total_loss = 0.0
        start_time = time.time()

        print(f"\n[INFO] Epoch {epoch+1}/{args.end_epoch} started...")

        for batch_idx, batch in enumerate(dl):
            optimiser.zero_grad()
            print(batch_idx)
    
   

    return


