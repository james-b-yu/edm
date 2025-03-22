#!/usr/bin/env python
"""
This script is used to perform all running
"""

import warnings

import os
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
from models.regularization_edm import RegularizationEDM
from qm9.visualise_samples import save_xyz_file,load_xyz_files, load_molecule_xyz, plot_data3d
import pickle

# from models.regularization_edm import RegularizationEDM

from os import path
import pickle
import wandb
from tqdm import tqdm
from args import args, parser
from tqdm.auto import tqdm

if __name__ == "__main__":
    torch.manual_seed(args.seed)

    print(args.sample_save_dir)
    
    if args.checkpoint is not None:
        try:
            with open(path.join(args.checkpoint, "args.pkl"), "rb") as f:
                args_disk = pickle.load(f)
                args = parser.parse_args(namespace=args_disk)
                
                # when loading checkpoint, we need to ensure that the sample_save_dir, sample_load_dir, and visual_save_dir are set correctly
                # they may not be included with the pickle depending on how the checkpoint was saved
                base_output_dir = os.path.join("outputs", args.extension, "samples")

                if args.sample_save_dir is None:
                    args.sample_save_dir = os.path.join(base_output_dir, "xyz")
                if args.sample_load_dir is None:
                    args.sample_load_dir = os.path.join(base_output_dir, "xyz")
                if args.visual_save_dir is None:
                    args.visual_save_dir = os.path.join(base_output_dir, "images")

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
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device),strict=False)  # always load the model as checkpoint
        
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
        samples = model.sample(num_molecules, batch_size, mol_sizes, mol_size_probs)


        if args.save_samples_xyz:
            dataset_info = get_dataset_info(remove_h=args.dataset == "qm9_no_h")
            output_path = args.sample_save_dir
            os.makedirs(output_path, exist_ok=True)

            for i, (coords, one_hot, charges) in enumerate(samples):
                coords = torch.tensor(coords).unsqueeze(0)         # [1, N, 3]
                one_hot = torch.tensor(one_hot).unsqueeze(0)        # [1, N, A]
                charges = torch.tensor(charges).unsqueeze(0)        # [1, N]

                save_xyz_file(
                    path=output_path,
                    one_hot=one_hot,
                    charges=charges,
                    positions=coords,
                    dataset_info=dataset_info,
                    id_from=i,
                    name="sample"
                )


    
        
        # number_molecules_stable = 0
        # number_atoms_stable = 0
        # number_atoms = 0
        
        # for s in range(len(samples)):
        #     print(f"molecule {s+1} has {len(samples[s][0])} atoms")
        #     number_atoms += len(samples[s][0])
        #     # xyz coords of each atom
            
        #     # print(f"xyz coords: {samples[0][0]}")
        #     coords = torch.tensor(samples[s][0], device=args.device)
        
        #     # one hot encoding of atom type H, C, O, N, F
        #     # print(f"one hot encoding of atoms: {samples[0][1]}")
        #     one_hot = torch.tensor(samples[s][1], device=args.device)
        
        #     # predicted valencies of each atom
        #     # print(f"charges: {samples[0][2]}")
        #     charges = torch.tensor(samples[s][2], device=args.device)

        #     node_mask = torch.ones(len(samples[s][0]), dtype=torch.bool, device=args.device)  # Shape: [num_atoms]

        #     if args.dataset == 'qm9':
        #         remove_h = False
        #     elif args.dataset == 'qm9_no_h':
        #         remove_h = True
            
        #     dataset_info = get_dataset_info(remove_h)
            
        #     # Ensure inputs are in the correct format
        #     one_hot = one_hot.unsqueeze(0)  # Add batch dimension
        #     charges = charges.unsqueeze(0).unsqueeze(-1)  # Add batch and last dimension
        #     coords = coords.unsqueeze(0)  # Add batch dimension
        #     node_mask = node_mask.unsqueeze(0)  # Add batch dimension
            
        #     atom_stability = compute_atom_stability(one_hot, charges, coords, node_mask, dataset_info)
        #     number_atoms_stable += atom_stability.sum().item()
        #     if number_atoms_stable == len(samples[s][0]):
        #         number_molecules_stable += 1
        #     print(f"number_atoms_stable: {number_atoms_stable}")

        # percentage_atoms_stable = (number_atoms_stable / number_atoms) * 100
        # percentage_molecules_stable = (number_molecules_stable / num_molecules) * 100
        # print(f"percentage_atoms_stable: {percentage_atoms_stable:.2f} %")
        # print(f"percentage_molecules_stable: {percentage_molecules_stable:.2f} %")
        
        
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
        
        # pass
    elif args.pipeline == "visualise":

        # Define your path and dataset
        xyz_path = args.sample_load_dir
        dataset_info = get_dataset_info(remove_h=args.dataset == "qm9_no_h")

        # Load files
        xyz_files = load_xyz_files(xyz_path, shuffle=False)

        for file in tqdm(xyz_files, desc="Visualising molecules"):
            coords, one_hot, _ = load_molecule_xyz(file, dataset_info)
            
            atom_types = torch.argmax(one_hot, dim=-1).numpy()
            centered_coords = coords - coords.mean(dim=0, keepdim=True)

            # Save each molecule as an image
            img_name = os.path.splitext(os.path.basename(file))[0] + ".png"
            img_path = os.path.join(args.visual_save_dir, img_name)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            plot_data3d(
                positions=centered_coords,
                atom_type=atom_types,
                dataset_info=dataset_info,
                save_path=img_path,
                spheres_3d=True,
                bg='black',
                alpha=0.6
            )

    
    else:
        raise NotImplementedError
else:
    raise RuntimeError