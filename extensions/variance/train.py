from argparse import Namespace
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from os import path, makedirs
import pickle
from wandb.wandb_run import Run

from data import EDMDataset, EDMDataloaderItem
from model import EGNNConfig
from utils.diffusion import Queue, gradient_clipping, scale_inputs

from .variance import VarianceDiffusion

def one_epoch(args: Namespace, epoch: int, split: Literal["train", "valid", "test"], model: VarianceDiffusion, gradnorm_queue: Queue, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    losses: list[float] = []
    dists: list[float] = []
    grad_norms: list[float] = []
    
    for idx, data in enumerate(pbar := tqdm(dataloaders[split], leave=False, unit="batch")):
        data: EDMDataloaderItem
        
        data.to_(args.device)
        
        s_coords, s_one_hot, s_charges = scale_inputs(data.coords, data.one_hot, data.charges)
        
        time_batch = torch.randint(low=0, high=args.num_steps + 1, size=(data.n_nodes.shape[0], ), dtype=torch.long, device=args.device)
        time = time_batch[data.expand_idx]  # random time steps repeated across atoms in each batch
        time_float = time.to(torch.float32)
        
        eps_coords = data.demean @ torch.randn_like(s_coords)
        eps_one_hot = torch.randn_like(s_one_hot)
        eps_charges = torch.randn_like(s_charges)
        
        alf = model.schedule["alpha"][time]
        alf_sq = model.schedule["alpha_squared"][time]
        alf_L_sq = model.schedule["alpha_L_squared"][time]
        sig = model.schedule["sigma"][time]
        sig_sq = model.schedule["sigma_squared"][time]
        bet_f_sq   = model.schedule["beta_squared"][time]       # the SQUARE of the forward transition variances  (note the first element is a dummy)
        
        z_coords = alf[:, None] * s_coords + sig[:, None] * eps_coords
        z_one_hot = alf[:, None] * s_one_hot + sig[:, None]* eps_one_hot
        z_charges = alf[:, None] * s_charges + sig[:, None]* eps_charges
        
        eps_features = torch.cat([eps_one_hot, eps_charges], dim=-1)
        z_features = torch.cat([z_one_hot, z_charges], dim=-1)
        
        pred_eps_coords, pred_eps_features = model.egnn(n_nodes=data.n_nodes,
                                                        coords=z_coords,
                                                        features=z_features,
                                                        edges=data.edges,
                                                        reduce=data.reduce,
                                                        demean=data.demean,
                                                        time=time_float)
        
        
        gamma_x = model.gamma_x(time)
        gamma_x_batch = model.gamma_x(time_batch)
        gamma_h = model.gamma_h(time)
        gamma_h_batch = model.gamma_h(time_batch)
        
        bet_b_batch = model.schedule["beta_sigma"][time_batch]   # backward transition variances (note the first element is a dummy)
        
        # For train likelihood, treat L0 like L12,3,4, etc.
        dim_coord = (data.n_nodes - 1.) * 3  # dimension of coordinate subspace for each molecule
        
        loss_coord = data.mean @ (((pred_eps_coords - eps_coords) ** 2 / gamma_x[:, None]).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
                   + dim_coord * (gamma_x_batch.log() - bet_b_batch.log() + bet_b_batch / gamma_x_batch)
        loss_features = data.mean @ (((pred_eps_features - eps_features) ** 2 / gamma_h).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
                   + (data.n_nodes[:, None] * (gamma_h_batch.log() - bet_b_batch[:, None].log() + bet_b_batch[:, None] / gamma_h_batch)).sum(dim=-1)
        
        total_loss = (loss_coord + loss_features)
        total_loss = total_loss.mean()
            
        ## calculate comparable losses
        with torch.no_grad():
            eps = torch.cat([eps_coords, eps_features], dim=-1)
            pred_eps = torch.cat([pred_eps_coords, pred_eps_features], dim=-1)
            
            loss_coord_compat = (1./3.) * 0.5 * (pred_eps_coords - eps_coords).norm(dim=-1) ** 2
            loss_features_compat = (1./model.egnn.config.features_d) * 0.5 * (pred_eps_features - eps_features).norm(dim=-1) ** 2
            total_loss_compat = data.mean @ (loss_coord_compat + loss_features_compat)
            total_loss_compat = total_loss_compat.mean()
            # total_loss_compat.backward()
        
        grad_norm = 0.
        if split == "train":
            total_loss.backward()
            if args.clip_grad:
                grad_norm = gradient_clipping(model, gradnorm_queue, max=args.max_grad_norm)
        
        pbar.set_description(f"Total loss: {total_loss:.2f} Compat loss: {total_loss_compat:.2f} Grad norm: {grad_norm:.2f}")
        
        if wandb_run is not None:
            wandb_run.log({
                f"{split}_batch_loss": total_loss,
                f"{split}_batch_avr_dist": total_loss_compat,
                f"{split}_batch_grad_norm": grad_norm
            })
        
        losses.append(float(total_loss))
        dists.append(float(total_loss_compat))
        grad_norms.append(float(grad_norm))
        
    return tuple(sum(l) / len(l) for l in [losses, dists, grad_norms])
    

def enter_train_valid_test_loop(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    
    for _, dl in dataloaders.items():
        assert(isinstance(dl.dataset, EDMDataset))
        features_d = dl.dataset.num_atom_classes + 1
    
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    model = VarianceDiffusion(egnn_config=EGNNConfig(
        features_d=features_d,
        node_attr_d=0,
        edge_attr_d=0,
        hidden_d=args.hidden_d,
        num_layers=args.num_layers,
    ), num_steps=args.num_steps, device=args.device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12
    )
    
    if args.checkpoint is not None:
        print(f"Loading model checkpoints using 'model.pth', 'optim.pth' located in {args.checkpoint}")
        model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))
        optim.load_state_dict(torch.load(path.join(args.checkpoint, "optim.pth"), map_location=args.device))
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        makedirs(path.join(args.out_dir, args.run_name), exist_ok=True)
        
        model.train()
        optim.zero_grad()
        loss, dist, grad_norm = one_epoch(args, epoch, "train", model, gradnorm_queue, dataloaders, wandb_run)
        optim.step()
        
        # now save things
        torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, "optim.pth"))
        torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, f"optim_epoch_{epoch}.pth"))
        
        torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, "model.pth"))
        torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, f"model_epoch_{epoch}.pth"))
        
        with open(path.join(args.out_dir, args.run_name, "args.pkl"), "wb") as f:
            pickle.dump(args, f)
        with open(path.join(args.out_dir, args.run_name, f"args_epoch_{epoch}.pkl"), "wb") as f:
            pickle.dump(args, f)
            
        if wandb_run is not None:
            wandb_run.log({
                "train_loss": loss,
                "train_dist": dist,
                "train_grad_norm": grad_norm,
                "epoch": epoch
            })