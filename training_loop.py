import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from tqdm import tqdm

from masked_model import MaskedEDM
from model import EDM
from model_config import get_config_from_args
from utils.diffusion import cosine_noise_schedule,polynomial_schedule

# from model import EGNN, EGNNConfig
from data import QM9Dataset, EDMDataloaderItem
from noise_schedule import default_noise_schedule
from utility import collate_fn, gradient_clipping, random_rotation
from losses import compute_loss_and_nll, compute_loss


def train_model(args,dataloader,log_file="logs/alt_3_training_log.csv",checkpoint_interval=5,run=1):
    
    # Set device
    device = args.device
    print(f"[INFO] Using device: {device}")

    if args.noise_schedule == "cosine":
        noise_schedule = cosine_noise_schedule(args.num_steps, device)
    elif args.noise_schedule == "polynomial":
        noise_schedule = polynomial_schedule(args.num_steps, device)
    else:
        raise ValueError(f"Invalid noise schedule: {args.noise_schedule}")
    
    print("[INFO] Noise schedule created.")

    # Initialize model
    config = get_config_from_args(args, dataloader.dataset.num_atom_types) 
    model = EDM(config) if args.use_non_masked else MaskedEDM(config)
    
    optimiser = optim.Adam(model.parameters(), lr=args.lr)


    # Assume `optimizer` is already created
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode="min",  # Typically "min" for reducing LR on loss plateau, or "max" for accuracy
    factor=args.scheduler_factor,  # Factor by which the LR is reduced (e.g., 0.5 means LR is halved)
    patience=args.scheduler_patience,  # Number of epochs with no improvement before reducing LR
    threshold=args.scheduler_threshold,  # Minimum relative improvement to reset patience
    min_lr=args.scheduler_min_lr,  # Minimum possible LR
)

    print(f"[INFO] Model Created.")


    print(f"[INFO] Dataset Loaded: {len(dataloader)} molecules in training set.")


    log_dir = os.path.dirname(log_file)  # Extracts the directory path ('logs/')

    if log_dir and not os.path.exists(log_dir):  # Check if directory exists
        os.makedirs(log_dir)  # Create directory if it doesn't exist

    # Now, create the log file
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch, Loss, Time(s)\n")

    

    for epoch in range(args.start_epoch, args.end_epoch):

        print(epoch)

        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"\n[INFO] Epoch {epoch+1}/{args.end_epoch} started...")
        
        for idx, data in enumerate(pbar := tqdm(dataloader)):

            optimiser.zero_grad()

            data.to_(dtype=torch.float32, device=args.device)
            batch_size = data["batch_size"]  
            time_int = torch.randint(low=0, high=args.num_steps + 1, size=(batch_size, ), device=args.device, dtype=torch.long)

            if args.use_non_masked:
                assert(isinstance(model, EDM))
                assert(isinstance(data, EDMDataloaderItem))
                (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = model.get_eps_and_predicted_eps(data, time_int=time_int)
            else:
                assert(isinstance(model, MaskedEDM))
                (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = model.get_eps_and_predicted_eps(data["positions"], data["one_hot"], data["charges"], time_int, data["node_mask"], data["edge_mask"])
            
            sq_coord_err = (eps_coord - pred_eps_coord) ** 2
            sq_feat_err  = (eps_feat - pred_eps_feat) ** 2
            
            mse = torch.concat([sq_coord_err, sq_feat_err], dim=-1).mean()
            pbar.set_description(f"Batch MSE {mse:.2f}")

            # Backward pass
            mse.backward()

            # Gradient clipping (if needed)
            if not args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimiser.step()

            # Accumulate total loss
            total_loss += mse.item()
        
        # End of epoch
        epoch_loss = total_loss / len(dataloader)
        elapsed_time = time.time() - start_time

        print(f"[INFO] Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

        # Log the epoch loss and time
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}, {epoch_loss:.4f}, {elapsed_time:.2f}\n")
        
        # Step the scheduler
        # ideally use validation loss here
        scheduler.step(epoch_loss)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            # checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            checkpoint_path = f"{args.out_dir}/{run}/epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"[INFO] Checkpoint saved at {checkpoint_path}")

    print("[INFO] Training completed.")
    
    # Save the final model
    final_model_dir = os.path.join("trained_model", str(run))
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    final_model_path = os.path.join(final_model_dir, "final_model.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss,
    }, final_model_path)
    print(f"[INFO] Final model saved at {final_model_path}")


           
# TODO: Implement the `validate_model` function
# TODO: Implement data augmentation (e.g., random rotation)
# TODO: Make outputs consistent with args.py

    
        

        
