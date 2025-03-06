import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import sys
import numpy as np

from model import EGNN, EGNNConfig
from data import QM9Dataset, EDMDataloaderItem
from noise_schedule import default_noise_schedule
from utility import collate_fn, gradient_clipping, random_rotation
from losses import compute_loss_and_nll, compute_loss


def train_edm(num_epochs=10, batch_size=64, learning_rate=1e-4, num_steps=1000, checkpoint_interval=1,log_file="alt_training_loss.txt"):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load full dataset
    dataset = QM9Dataset(use_h=True, split="train")

    # Set seed for reproducibility
    np.random.seed(42)
    subset_indices = np.random.choice(len(dataset), int(0.1 * len(dataset)), replace=False)

    # Create DataLoader with subset sampler
    sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Load dataset
    # dataset = QM9Dataset(use_h=True, split="train")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers= 4, pin_memory=True)
    
    print(f"[INFO] Dataset Loaded: {len(sampler)} molecules in training set.")

    # Model configuration
    config = EGNNConfig(
        features_d=5,  
        node_attr_d=0,  
        edge_attr_d=0,  
        hidden_d=256,   
        num_layers=9    
    )

    # Initialize model
    model = EGNN(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.MSELoss()

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  

    # Define noise scheduler
    noise_schedule = default_noise_schedule(num_steps, device)

    gradnorm_queue = torch.tensor([], dtype=torch.float32, device=device)


    # Create log file (if it doesn't exist)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch, Loss, Time(s)\n")  # Write header
    
    print("[INFO] Noise schedule created.")

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"\n[INFO] Epoch {epoch+1}/{num_epochs} started...")
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move batch data to device
            batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}

            batch["coords"] = random_rotation(batch["coords"], batch["n_nodes"]).detach()            
            
            # Debug tensor shapes
            if batch_idx == 0:
                print(f"[DEBUG] Batch {batch_idx} tensor shapes:")
                for key, val in batch.items():
                    print(f"  {key}: {val.shape}")

            # # Sample random timestep
            # t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
            # time_tensor = t / num_steps  
            
            # num_atoms_total = batch["coords"].shape[0]  
            # time_expanded = time_tensor.repeat_interleave(batch["n_nodes"]).reshape(num_atoms_total, 1)

            # # Retrieve noise scaling factors
            # alpha_t = noise_schedule["alpha"][t].unsqueeze(-1)  
            # sigma_t = noise_schedule["sigma"][t].unsqueeze(-1)  

            # alpha_t_expanded = alpha_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)
            # sigma_t_expanded = sigma_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)

            # # Generate noise
            # noise_coords = torch.randn_like(batch["coords"], dtype=torch.float32)
            # noise_features = torch.randn_like(batch["features"], dtype=torch.float32)

            # # Apply diffusion process
            # zt_coords = alpha_t_expanded * batch["coords"] + sigma_t_expanded * noise_coords
            # zt_features = alpha_t_expanded * batch["features"] + sigma_t_expanded * noise_features

            # # Predict noise using the model
            # predicted_coords, predicted_features = model(
            #     batch["n_nodes"], zt_coords, zt_features, batch["edges"], 
            #     batch["reduce"], batch["demean"], time_expanded
            # )

            # # Compute loss (normalized)
            # loss_coords = criterion(predicted_coords, noise_coords) / batch["coords"].shape[0]
            # loss_features = criterion(predicted_features, noise_features) / batch["features"].shape[0]
            # loss = loss_coords + loss_features

            # # Compute Loss using paper's function
            # nll, reg_term, mean_abs_z = compute_loss_and_nll(
            #     model, zt_coords, zt_features, batch["n_nodes"], batch["edges"], time_expanded
            # )

            # loss = nll + 0.01 * reg_term

            # Compute loss
            loss, loss_coords, loss_features = compute_loss(model, batch, noise_schedule)

            # Check if loss is NaN or infinite
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] NaN detected in loss at epoch {epoch+1}, batch {batch_idx}. Exiting...")
                sys.exit(1)  # Exit program with error code 1


            # Debugging loss values
            if batch_idx % 10 == 0:
                print(f"[DEBUG] Batch {batch_idx}: Loss = {loss.item():.6f}")

            # Backpropagation
            loss.backward()

            # Apply gradient clipping
            grad_norm, gradnorm_queue = gradient_clipping(model, gradnorm_queue)

            # Debugging gradients
            if batch_idx % 50 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"[DEBUG] Gradients in {name}: Mean={param.grad.mean().item():.6f}")

            optimizer.step()
            total_loss += loss.item()

            epoch_time = time.time() - start_time
            avg_loss = total_loss / len(dataloader)

        # update learning rate
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} completed. Loss: {total_loss/len(dataloader):.4f} (Time: {epoch_time:.2f}s)")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1}, {avg_loss:.6f}, {epoch_time:.2f}\n")  # Save loss in file


        # Save model checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"alt_checkpoint_epoch_{epoch+1}.pth")
            print(f"[INFO] Checkpoint saved: alt_checkpoint_epoch_{epoch+1}.pth")

    # Final model save
    torch.save(model.state_dict(), "trained_edm.pth")
    print("[INFO] Training complete. Model saved as trained_edm.pth")


def validate_edm(batch_size=64, num_steps=1000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset (use validation set)
    dataset = QM9Dataset(use_h=True, split="valid")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Load trained model
    config = EGNNConfig(features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256, num_layers=9)
    model = EGNN(config).to(device)
    model.load_state_dict(torch.load("trained_edm.pth"))
    model.eval()  # Set model to eval mode

    noise_schedule = default_noise_schedule(num_steps, device)

    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}

            t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
            time_tensor = t / num_steps
            num_atoms_total = batch["coords"].shape[0]
            time_expanded = time_tensor.repeat_interleave(batch["n_nodes"]).reshape(num_atoms_total, 1)

            # Predict noise using model
            predicted_coords, predicted_features = model(
                batch["n_nodes"], batch["coords"], batch["features"], batch["edges"],
                batch["reduce"], batch["demean"], time_expanded
            )

            # Compute validation loss
            loss_coords = F.mse_loss(predicted_coords, batch["coords"])
            loss_features = F.mse_loss(predicted_features, batch["features"])
            loss = loss_coords + loss_features
            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"[INFO] Validation Batch {batch_idx}: Loss = {loss.item():.6f}")

    print(f"[INFO] Validation Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    print("[INFO] Starting Training")
    train_edm(num_epochs=75, batch_size=64, learning_rate=1e-4, num_steps=1000)
    # print("[INFO] Starting Validation")
    # validate_edm()
