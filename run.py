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
from stability_unique_valid import compute_stability_unique_and_valid, check_stability
import multiprocessing as mp
from configs.datasets_config import get_dataset_info
import pickle

# from models.regularization_edm import RegularizationEDM

from os import path
import pickle
import wandb
from args import args, parser
from tqdm.auto import tqdm

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
    elif args.extension == "regularization":
        Model = RegularizationEDM
    else:
        raise NotImplementedError
    
    # just for KJ
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    assert hasattr(dataloaders["train"].dataset, "num_atom_types")
    model = Model(get_config_from_args(args, dataloaders["train"].dataset.num_atom_types))  # type: ignore
    
    if args.checkpoint is not None:                    
        print(f"Loading model checkpoint located in {args.checkpoint}")
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))  # always load the model as checkpoint
        
        # just for KJ as on a Mac and cannot load CUDA checkpoints directly
        map_location = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=map_location))
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
        
        
        ########################## sampling section ##########################################################
        # avoids running out of memory for larger samples
        all_samples = []
        model.to(args.device)

        with torch.no_grad():
            for i in tqdm(range(0, num_molecules, batch_size), desc="Sampling"):
                current_batch_size = min(batch_size, num_molecules - i)
                batch_samples = model.sample(current_batch_size, batch_size, mol_sizes, mol_size_probs)
                all_samples.extend(batch_samples)

        samples = all_samples
        
        with open("samples1_testing_again.pkl", "wb") as f:
            pickle.dump(samples, f)
        #########################################################################################################

        ########################## evaluation section ##########################################################
        # Load list from file
        import os
        print(os.path.getsize("samples1_testing_again.pkl"))  # Should be > 0

        with open("samples1_testing_again.pkl", "rb") as f:
            samples = pickle.load(f)


        dataset_info = get_dataset_info(remove_h=False)
        print(f"dataset_info: {dataset_info}")
        
        percentage_atom_stability, percentage_molecule_stability, validity_percentage, valid_and_unique_percentage, res = compute_stability_unique_and_valid(samples, args)

        print(f"results molecule stable, sample of 10 (T/F, number_stable_atoms, total_atoms_in_molecule): {res[0:10]}")
        print(f"percentage_atoms_stable: {percentage_atom_stability:.4f} %")
        print(f"percentage_molecules_stable: {percentage_molecule_stability:.4f} %")
        print(f"valid: {validity_percentage:.4f}")
        print(f"valid and unique: {valid_and_unique_percentage:.4f}")
        
        #########################################################################################################
        
        pass
    else:
        raise NotImplementedError
else:
    raise RuntimeError