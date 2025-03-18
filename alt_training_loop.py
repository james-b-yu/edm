import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import numpy as np
import copy

from models.base import EGNN, EGNNConfig
from data import QM9Dataset
from noise_schedule import default_noise_schedule
from utility import collate_fn, gradient_clipping, random_rotation
from qm9 import losses  # ✅ Import correct loss function
from diffusion_utils import EMA  # ✅ Import EMA for stability


def train_edm(num_epochs=10, batch_size=64, learning_rate=1e-4, num_steps=1000, 
              checkpoint_interval=1, log_file="training_loss.txt", checkpoint_path=None):
    
    # ✅ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ✅ Load dataset (Subset: 10% for faster testing)
    dataset = QM9Dataset(use_h=True, split="train")
    np.random.seed(42)
    subset_indices = np.random.choice(len(dataset), int(0.1 * len(dataset)), replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(subset_indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"[INFO] Dataset Loaded: {len(sampler)} molecules in training set.")

    # ✅ Model configuration
    config = EGNNConfig(features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256, num_layers=9)
    model = EGNN(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # ✅ Learning Rate Scheduler
    noise_schedule = default_noise_schedule(num_steps, device)

    # ✅ Initialize EMA for stability
    ema_decay = 0.999
    model_ema = copy.deepcopy(model)
    ema = EMA(ema_decay)

    # ✅ Initialize Gradient Norm Queue
    gradnorm_queue = torch.tensor([], dtype=torch.float32, device=device)

    # ✅ Resume Training if Checkpoint Exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        model_ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[INFO] Resumed training from epoch {start_epoch}")

    # ✅ Create log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch, Loss, Time(s)\n")

    print("[INFO] Noise schedule created.")

    # ✅ Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"\n[INFO] Epoch {epoch+1}/{num_epochs} started...")
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # ✅ Move batch data to device
            batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}

            # ✅ Apply Random Rotation
            batch["coords"] = random_rotation(batch["coords"], batch["n_nodes"]).detach()            
            
            # ✅ Sample random timestep
            t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
            time_tensor = t / num_steps  
            
            num_atoms_total = batch["coords"].shape[0]  
            time_expanded = time_tensor.repeat_interleave(batch["n_nodes"]).reshape(num_atoms_total, 1)

            # ✅ Retrieve noise scaling factors
            alpha_t = noise_schedule["alpha"][t].unsqueeze(-1)  
            sigma_t = noise_schedule["sigma"][t].unsqueeze(-1)  

            alpha_t_expanded = alpha_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)
            sigma_t_expanded = sigma_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)

            # ✅ Generate noise
            noise_coords = torch.randn_like(batch["coords"], dtype=torch.float32)
            noise_features = torch.randn_like(batch["features"], dtype=torch.float32)

            # ✅ Apply diffusion process
            zt_coords = alpha_t_expanded * batch["coords"] + sigma_t_expanded * noise_coords
            zt_features = alpha_t_expanded * batch["features"] + sigma_t_expanded * noise_features

            # ✅ Compute Loss using correct function
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
                model, zt_coords, zt_features, batch["n_nodes"], batch["edges"]
            )
            loss = nll + 0.01 * reg_term

            # ✅ Debugging loss values
            if batch_idx % 10 == 0:
                print(f"[DEBUG] Batch {batch_idx}: Loss = {loss.item():.6f}")

            # ✅ Backpropagation & Gradient Clipping
            loss.backward()
            grad_norm, gradnorm_queue = gradient_clipping(model, gradnorm_queue)

            optimizer.step()

            # ✅ Apply EMA Update
            ema.update_model_average(model_ema, model)

            total_loss += loss.item()

        scheduler.step()  # ✅ Adjust learning rate

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)

        print(f"[INFO] Epoch {epoch+1}/{num_epochs} completed. Loss: {avg_loss:.4f} (Time: {epoch_time:.2f}s)")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1}, {avg_loss:.6f}, {epoch_time:.2f}\n")  # Save loss in file

        # ✅ Save Checkpoint Periodically
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": model_ema.state_dict(),
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"[INFO] Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    # ✅ Final Model Save
    torch.save(model.state_dict(), "trained_edm.pth")
    print("[INFO] Training complete. Model saved as trained_edm.pth")


if __name__ == "__main__":
    print("[INFO] Starting Training")
    train_edm(num_epochs=10, batch_size=64, learning_rate=1e-4, num_steps=1000, checkpoint_path="checkpoint_epoch_5.pth")
