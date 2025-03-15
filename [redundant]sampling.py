import torch
import argparse
import os
import numpy as np
from masked_model import MaskedEDM
from model_config import get_config_from_args
from masked_data import get_masked_qm9_dataloader
from noise_schedule import PredefinedNoiseSchedule


def sample(args):

    print("Sampling")

    device = args.device
    print(f"[INFO] Using device: {device}")

    # Load dataset (for shape and scaling reference)
    dataloader = get_masked_qm9_dataloader(
        use_h=args.dataset == "qm9", split="test", batch_size=args.batch_size
    )
    config = get_config_from_args(args, dataloader.dataset.num_atom_types)

    # Load model
    model = MaskedEDM(config)
    
    checkpoint_path = args.checkpoint
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        print("Loaded Checkpoint")
        # print(checkpoint.keys())
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError("Checkpoint file not found!")
    
    model.eval().to(device)
    
    # Load noise schedule
    noise_schedule = PredefinedNoiseSchedule(args.noise_schedule, args.num_steps, precision=1e-4)

    # Sample from the model
    # num_samples = args.num_samples
    num_samples = 1

    # batch_size = args.batch_size

    batch_size = 1


    sample_shape = (batch_size, dataloader.dataset.max_nodes, 3)
    
    generated_molecules = []
    with torch.no_grad():
        for i in range(num_samples // batch_size):

            print(i)
            
            # Initialize noise
            coord = torch.randn(sample_shape, device=device)
            one_hot = torch.randn((batch_size, dataloader.dataset.max_nodes, dataloader.dataset.num_atom_types), device=device)
            charge = torch.randn((batch_size, dataloader.dataset.max_nodes, 1), device=device)
            node_mask = torch.ones((batch_size, dataloader.dataset.max_nodes, 1), device=device)
            edge_mask = torch.ones((batch_size, dataloader.dataset.max_nodes ** 2, 1), device=device)
            
            for t in reversed(range(args.num_steps)):
                print(t)
                time_int = torch.full((batch_size,), t, device=device, dtype=torch.long)
                time_frac = time_int / args.num_steps
                pred_eps_coord, pred_eps_feat = model.egnn(coord, one_hot, time_frac, node_mask, edge_mask)
                
                alpha_t = noise_schedule.gamma[time_int][:, None, None].exp()
                sigma_t = torch.sqrt(1 - alpha_t**2)
                
                coord = (coord - sigma_t * pred_eps_coord) / alpha_t
                one_hot = (one_hot - sigma_t * pred_eps_feat) / alpha_t
                
                if t > 0:
                    noise = torch.randn_like(coord)
                    coord += sigma_t * noise
            
            generated_molecules.append(coord.cpu().numpy())
    
    generated_molecules = np.concatenate(generated_molecules, axis=0)
    np.save("generated_molecules.npy", generated_molecules)
    print(f"[INFO] Generated molecules saved as generated_molecules.npy")


if __name__ == "__main__":

    print("Start Sampler Script")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="trained_edm.pth", required=False, help="Path to the trained model checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of molecules to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for sampling")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset used for training")
    parser.add_argument("--noise-schedule", type=str, default="polynomial", help="Noise schedule")
    
    args = parser.parse_args()

    print("args: ")
    print(args)

    sample(args)
