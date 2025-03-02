import torch
from model import EGNN,EGNNConfig  # Model
from data import EDMDataloaderItem  # For handling molecule data
import torch.nn.functional as F

class Sampler():

    def __init__(self, model, num_steps=1000, noise_schedule=None,device="cuda"):

        # Ensure model is in eval mode
        self.model = model.to(device).eval()  
        
        self.num_steps = num_steps
        self.device = device


        # if not given a schedule, create one
        # self.noise_schedule = noise_schedule if noise_schedule else self.default_noise_schedule()

        # alternative cosine noise
        self.noise_schedule = self._cosine_noise_schedule()

    # Creates a default noise schedule if not specified
    def default_noise_schedule(self):
        
        t = torch.linspace(0, 1, self.num_steps)
        
        # schedule
        alpha_t = torch.sqrt(1 - t**2) 
        
        # Noise
        sigma_t = torch.sqrt(t**2) 
        
        return {"alpha": alpha_t, "sigma": sigma_t}
    
    def _cosine_noise_schedule(self):
        t = torch.linspace(0, 1, self.num_steps, device=self.device)
        alpha_t = torch.cos((t + 0.008) / 1.008 * (torch.pi / 2)) ** 2
        sigma_t = torch.sqrt(1 - alpha_t**2)
        return {"alpha": alpha_t, "sigma": sigma_t}    
    
    @torch.no_grad()
    def sample(self, num_atoms=10):

        print("Samping Started")

        # Initialize random noise for coordinates, and for features with 5 possible atoms
        coords = torch.randn((num_atoms, 3), device=self.device)
        features = F.one_hot(torch.randint(0, 5, (num_atoms,)), num_classes=5).float().to(self.device)

        # Define molecular graph structure
        n_nodes = torch.tensor([num_atoms], dtype=torch.int64, device=self.device)
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
            alpha_t = self.noise_schedule["alpha"][t]
            sigma_t = self.noise_schedule["sigma"][t]

            # Generate random noise and enforce zero center of gravity
            epsilon_x = torch.randn_like(coords)
            epsilon_x -= epsilon_x.mean(dim=0, keepdim=True)
            epsilon_h = torch.randn_like(features)
            
            # Apply reverse diffusion update
            coords = (1 / alpha_t) * (coords - sigma_t * predicted_coords) + sigma_t * epsilon_x
            features = (1 / alpha_t) * (features - sigma_t * predicted_features) + sigma_t * epsilon_h

        print("Sampling Finished")
        
        return coords.cpu().numpy(), features.cpu().numpy()
    
# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"


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
model.eval()

# Sample a new molecule
sampler = Sampler(model)
generated_coords, generated_features = sampler.sample(num_atoms=20)

# Save or visualize the result
print("Generated Coordinates:", generated_coords)
print("Generated Features:", generated_features)