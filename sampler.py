import torch
from model import EGNN,EGNNConfig  # Model
from data import EDMDataloaderItem  # For handling molecule data
from utils.diffusion import cosine_noise_schedule, default_noise_schedule # For creating noise schedules
import torch.nn.functional as F

from noise_schedule import default_noise_schedule, cosine_noise_schedule

class Sampler():

    def __init__(self, model, num_steps=1000, noise_schedule=None,device="cuda"):

        # Ensure model is in eval mode
        self.model = model.to(device).eval()  
        
        self.num_steps = num_steps
        self.device = device


        # if not given a schedule, create one
        # self.noise_schedule = noise_schedule if noise_schedule else self.default_noise_schedule(self.num_steps, self.device)

        # alternative cosine noise
        self.noise_schedule = cosine_noise_schedule(self.num_steps, self.device)
    
    @torch.no_grad()
    def sample(self, num_atoms=10):

        print("Samping Started")

        # Initialize random noise for coordinates, and for features with 5 possible atoms
        coords = torch.randn((num_atoms, 3), device=self.device)
        features = F.one_hot(torch.randint(0, 5, (num_atoms,)), num_classes=5).float().to(self.device)

        # print("coords: ")
        # print(coords)

        # print("features: ")
        # print(features)

        # Define molecular graph structure
        n_nodes = torch.tensor([num_atoms], dtype=torch.long, device=self.device)
        edges = torch.cartesian_prod(torch.arange(num_atoms, device=self.device), torch.arange(num_atoms, device=self.device))
        
        # reduce_matrix
        num_edges = edges.shape[0]
        reduce_matrix = torch.zeros((num_atoms, num_edges), device=self.device)
        
        for i in range(num_edges):
            reduce_matrix[edges[i, 0], i] = 1.0

        # demean_matrix
        demean_matrix = torch.eye(num_atoms, device=self.device) - (torch.ones((num_atoms, num_atoms), device=self.device) / num_atoms)

        # Reverse diffusion process
        for t in reversed(range(1, self.num_steps)):
            print("t: " +str(t))
            
            time_tensor = torch.tensor([t / self.num_steps], dtype=torch.float32, device=self.device)

            # Predict noise using the model
            predicted_coords, predicted_features = self.model(n_nodes, coords, features, edges, reduce_matrix, demean_matrix, time_tensor)

            # Get noise schedule values
            alpha_t = torch.clamp(self.noise_schedule["alpha"][t], min=1e-5)
            sigma_t = torch.clamp(self.noise_schedule["sigma"][t], min=1e-5)

            # Generate random noise and enforce zero center of gravity
            epsilon_x = torch.randn_like(coords)
            epsilon_x -= epsilon_x.mean(dim=0, keepdim=True)
            epsilon_h = torch.randn_like(features)
            
            # Apply reverse diffusion update

            eps = 1e-7  # Small constant to prevent division by very small numbers
            coords = (1 / (alpha_t)) * (coords - sigma_t * predicted_coords) + sigma_t * epsilon_x
            features = (1 / (alpha_t)) * (features - sigma_t * predicted_features) + sigma_t * epsilon_h
            features = F.one_hot(features.argmax(dim=-1), num_classes=5).float()


            # coords = (1 / alpha_t) * (coords - sigma_t * predicted_coords) + sigma_t * epsilon_x
            # features = (1 / alpha_t) * (features - sigma_t * predicted_features) + sigma_t * epsilon_h

        print("Sampling Finished")
        
        return coords.cpu().numpy(), features.cpu().numpy()
    

    # alternative samplign function that takes into consideration masking
    # max_n_nodes should match dataset's largest molecule
    @torch.no_grad()
    def sample_masked(self, num_atoms=10, max_n_nodes=29):

        print("Sampling Started (masked)")

        # Create a node_mask (1 for real atoms, 0 for padding)
        node_mask = torch.zeros(max_n_nodes, device=self.device)
        
        # Mark real atoms
        node_mask[:num_atoms] = 1  

        
        # Initialize coordinates and features with padding
        
        # All zeros by default
        coords = torch.zeros((max_n_nodes, 3), device=self.device)  

        # Noise for real atoms
        coords[:num_atoms, :] = torch.randn((num_atoms, 3), device=self.device)  

        features = torch.zeros((max_n_nodes, 5), device=self.device)  
        features[:num_atoms, :] = F.one_hot(torch.randint(0, 5, (num_atoms,)), num_classes=5).float().to(self.device)

        # Define molecular graph structure
        n_nodes = torch.tensor([num_atoms], dtype=torch.long, device=self.device)
        edges = torch.cartesian_prod(torch.arange(max_n_nodes, device=self.device), torch.arange(max_n_nodes, device=self.device))

        # reduce_matrix
        num_edges = edges.shape[0]
        reduce_matrix = torch.zeros((max_n_nodes, num_edges), device=self.device)

        for i in range(num_edges):
            reduce_matrix[edges[i, 0], i] = 1.0

        # demean_matrix
        demean_matrix = torch.eye(max_n_nodes, device=self.device) - (torch.ones((max_n_nodes, max_n_nodes), device=self.device) / max_n_nodes)

        # Reverse diffusion process
        for t in reversed(range(1, self.num_steps)):
            print(f"t: {t}")

            time_tensor = torch.tensor(t / self.num_steps, dtype=torch.float32, device=self.device)

            # Predict noise using the model
            predicted_coords, predicted_features = self.model(n_nodes, coords, features, edges, reduce_matrix, demean_matrix, time_tensor)

            # Get noise schedule values
            alpha_t = self.noise_schedule["alpha"][t]
            sigma_t = self.noise_schedule["sigma"][t]

            # Generate random noise (only for real atoms)
            # Expand for broadcasting
            epsilon_x = torch.randn_like(coords) * node_mask.unsqueeze(-1)  
            # Zero center of gravity
            epsilon_x -= epsilon_x.mean(dim=0, keepdim=True)  

            epsilon_h = torch.randn_like(features) * node_mask.unsqueeze(-1)

            # Apply reverse diffusion update (only real atoms)
            coords = (1 / alpha_t) * (coords - sigma_t * predicted_coords) + sigma_t * epsilon_x
            features = (1 / alpha_t) * (features - sigma_t * predicted_features) + sigma_t * epsilon_h
            

            # Ensure padding atoms remain zero
            # Expand for 3D coords
            coords *= node_mask.unsqueeze(-1)
            # Expand for feature dim
            features *= node_mask.unsqueeze(-1)

        print("Sampling Finished")

        # Return only real atoms
        return coords[:num_atoms].cpu().numpy(), features[:num_atoms].cpu().numpy()  

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"


# Use this when there is an actually trained model to load parameters from
# model = EGNN.load_state_dict(torch.load("model_checkpoint.pth", map_location=device))


# Initialize a random EGNN model with dummy parameters
dummy_config = EGNNConfig(
    features_d=5,      # Number of atom types
    node_attr_d=0,     # Additional attributes per node
    edge_attr_d=0,     # Additional attributes per edge
    hidden_d=256,       # Hidden layer size
    num_layers=9       # Number of EGNN layers
)

model = EGNN(dummy_config)

model.to(device)

# Load trained weights
model.load_state_dict(torch.load("checkpoint_epoch_44.pth", map_location=device))

model.eval()

# Sample a new molecule
sampler = Sampler(model,num_steps = 4)

# try:
    # generated_coords, generated_features = sampler.sample_masked(num_atoms=10, max_n_nodes=29)
# except Exception as e:
#     print(f"Error in masked sampling: {e}")

generated_coords, generated_features = sampler.sample(num_atoms=2)


# Save or visualize the result
print("Generated Coordinates:", generated_coords)
print("Generated Features:", generated_features)
print("Center of Gravity:", generated_coords.mean(axis=0))
