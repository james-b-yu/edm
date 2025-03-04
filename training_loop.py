import torch
import torch.optim as optim
import torch.nn.functional as F

from model import EGNN, EGNNConfig  # Import the model
from data import QM9Dataset, EDMDataloaderItem  # Import dataset
from noise_schedule import default_noise_schedule
from utility import collate_fn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = QM9Dataset(use_h=True, split="train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Model configuration
config = EGNNConfig(
    features_d=5,      # Number of atom types
    node_attr_d=0,     # Additional node attributes
    edge_attr_d=0,     # Additional edge attributes
    hidden_d=256,      # Hidden layer size
    num_layers=9       # Number of EGNN layers
)

# Initialize model
model = EGNN(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Define noise scheduler
num_steps = 1000 
noise_schedule  = default_noise_schedule(num_steps, device)

# Training settings
num_epochs = 10  # Increase training epochs
checkpoint_interval = 2  # Save every 2 epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Ensure batch data is moved to device
        batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}

        # Sample a random timestep
        t = torch.randint(1, num_steps, (batch["n_nodes"].shape[0],), device=device)
        time_tensor = t / num_steps  # Normalize time

        # Retrieve noise scaling factors
        alpha_t = noise_schedule["alpha"][t].unsqueeze(-1)  # (64, 1)
        sigma_t = noise_schedule["sigma"][t].unsqueeze(-1)  # (64, 1)

        # Expand scaling factors for all atoms in the batch
        num_atoms_total = batch["coords"].shape[0]  # Sum of atoms across all molecules
        alpha_t_expanded = alpha_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)
        sigma_t_expanded = sigma_t.repeat_interleave(batch["n_nodes"], dim=0).reshape(num_atoms_total, 1)

        # Generate noise
        noise_coords = torch.randn_like(batch["coords"])
        noise_features = torch.randn_like(batch["features"])

        # Apply diffusion process
        zt_coords = alpha_t_expanded * batch["coords"] + sigma_t_expanded * noise_coords
        zt_features = alpha_t_expanded * batch["features"] + sigma_t_expanded * noise_features

        # Predict noise using the model
        predicted_coords, predicted_features = model(
            batch["n_nodes"], zt_coords, zt_features, batch["edges"], 
            batch["reduce"], batch["demean"], time_tensor
        )

        # Compute loss
        loss_coords = criterion(predicted_coords, noise_coords)
        loss_features = criterion(predicted_features, noise_features)
        loss = loss_coords + loss_features

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model checkpoint periodically
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

# Final model save
torch.save(model.state_dict(), "trained_edm.pth")
print("Training complete. Model saved.")
