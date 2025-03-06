import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import os
import copy

from model import EGNN, EGNNConfig
from data import QM9Dataset
from noise_schedule import default_noise_schedule
from utility import collate_fn
from losses import *  # ✅ Use paper's loss function
from diffusion_utils import EMA  # ✅ EMA for stability
from utils import gradient_clipping, random_rotation  # ✅ Augmentation & clipping

# Enable faster GPU optimizations
cudnn.benchmark = True

# Use torch.compile() for PyTorch 2.0+ (optional)
USE_TORCH_COMPILE = True  

def train_edm(num_epochs=10, batch_size=64, learning_rate=1e-4, num_steps=1000, 
              checkpoint_interval=1, log_file="training_loss.txt", checkpoint_path=None):
    """Train the Equivariant Diffusion Model (EDM) with best practices from the paper"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset efficiently
    dataset = QM9Dataset(use_h=True, split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    # Model configuration
    config = EGNNConfig(features_d=5, hidden_d=256, num_layers=9)
    model = EGNN(config).to(device)

    # Apply torch.compile() for PyTorch 2.0+
    if USE_TORCH_COMPILE:
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # ✅ Adaptive learning rate
    scaler = torch.cuda.amp.GradScaler()  # ✅ Mixed precision

    # Noise scheduler
    noise_schedule = default_noise_schedule(num_steps, device)

    # ✅ Initialize EMA for training stability
    ema_decay = 0.999
    model_ema = copy.deepcopy(model) if ema_decay > 0 else None
    ema = EMA(ema_decay) if ema_decay > 0 else None

    # ✅ Track starting epoch
    start_epoch = 0

    # ✅ Check if there's a checkpoint to resume from
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if model_ema:
            model_ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[INFO] Resumed training from epoch {start_epoch}")

    # ✅ Create log file (if it doesn't exist)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch, Loss, Time(s)\n")  # Write header

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}

            # ✅ Apply random rotation augmentation
            batch["coords"] = random_rotation(batch["coords"]).detach()

            t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
            time_tensor = t / num_steps

            # ✅ Use mixed precision
            with torch.cuda.amp.autocast():
                nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
                    model, batch["coords"], batch["features"], batch["n_nodes"], batch["edges"]
                )
                loss = nll + 0.01 * reg_term  # ✅ Regularization term

            scaler.scale(loss).backward()

            # ✅ Apply gradient clipping
            gradient_clipping(model)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # ✅ Apply EMA update
            if ema:
                ema.update_model_average(model_ema, model)

        scheduler.step()  # ✅ Adjust learning rate

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)

        # ✅ Print and log loss
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} completed. Loss: {avg_loss:.6f} (Time: {epoch_time:.2f}s)")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1}, {avg_loss:.6f}, {epoch_time:.2f}\n")  # Save loss in file

        # ✅ Save checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "ema_state_dict": model_ema.state_dict() if ema else None
            }
            checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint_data, checkpoint_filename)
            print(f"[INFO] Checkpoint saved: {checkpoint_filename}")

    # ✅ Final model save
    torch.save(model.state_dict(), "trained_edm.pth")
    print("[INFO] Training complete. Model saved as trained_edm.pth")


if __name__ == "__main__":
    print("[INFO] Starting Training")
    train_edm()
