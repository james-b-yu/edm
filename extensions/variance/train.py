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
from utils.diffusion import Queue, gradient_clipping, scale_features

from .variance import VarianceDiffusion

def one_epoch(args: Namespace, epoch: int, split: Literal["train", "valid", "test"], model: VarianceDiffusion, gradnorm_queue: Queue, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    losses: list[float] = []
    dists: list[float] = []
    grad_norms: list[float] = []
    
    vlbs: list[float] = []
    
    for idx, data in enumerate(pbar := tqdm(dataloaders[split], leave=False, unit="batch")):
        data: EDMDataloaderItem
        
        data.to_(args.device)        
        compat_loss, dist = model.calculate_loss(args=args, split=split, data=data)
        
        # pred_eps_coords, pred_eps_features = model.egnn()
        
        
        # gamma_x = model.gamma_x(time)
        # gamma_x_batch = model.gamma_x(time_batch)
        # gamma_h = model.gamma_h(time)
        # gamma_h_batch = model.gamma_h(time_batch)
        
        # bet_b_batch = model.schedule["beta_sigma"][time_batch]   # backward transition variances (note the first element is a dummy)
        
        # # For train likelihood, treat L0 like L12,3,4, etc.
        # dim_coord = (data.n_nodes - 1.) * 3  # dimension of coordinate subspace for each molecule
        
        # loss_coord = data.batch_mean @ (((pred_eps_coords - eps_coords) ** 2 / gamma_x[:, None]).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
        #            + dim_coord * (gamma_x_batch.log() - bet_b_batch.log() + bet_b_batch / gamma_x_batch)
        # loss_features = data.batch_mean @ (((pred_eps_features - eps_features) ** 2 / gamma_h).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
        #            + (data.n_nodes[:, None] * (gamma_h_batch.log() - bet_b_batch[:, None].log() + bet_b_batch[:, None] / gamma_h_batch)).sum(dim=-1)
        
        # total_loss = (loss_coord + loss_features)
        # total_loss = total_loss.mean()
            
        # ## calculate comparable losses
        # with torch.no_grad():
        #     eps = torch.cat([eps_coords, eps_features], dim=-1)
        #     pred_eps = torch.cat([pred_eps_coords, pred_eps_features], dim=-1)
            
        #     loss_coord_compat = (1./3.) * 0.5 * (pred_eps_coords - eps_coords).norm(dim=-1) ** 2
        #     loss_features_compat = (1./model.egnn.config.features_d) * 0.5 * (pred_eps_features - eps_features).norm(dim=-1) ** 2
        #     total_loss_compat = data.batch_mean @ (loss_coord_compat + loss_features_compat)
        #     total_loss_compat = total_loss_compat.mean()
        #     # total_loss_compat.backward()
        
        grad_norm = 0.
        if split == "train":
            dist.backward()
            if args.clip_grad:
                grad_norm = gradient_clipping(model, gradnorm_queue, max=args.max_grad_norm)
        
        pbar.set_description(f"({split}) Total loss: {dist:.2f} Compat loss: {compat_loss:.2f} Grad norm: {grad_norm:.2f}")
        
        if wandb_run is not None:
            wandb_run.log({
                f"{split}_batch_compat_loss": compat_loss,
                f"{split}_batch_avr_dist": dist,
                f"{split}_batch_grad_norm": grad_norm
            })
        
        losses.append(float(compat_loss))
        dists.append(float(dist))
        grad_norms.append(float(grad_norm))
        
    return tuple(sum(l) / len(l) for l in [losses, dists, grad_norms])
    

def enter_train_valid_test_loop(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run, no_train=False):
    
    for _, dl in dataloaders.items():
        assert(isinstance(dl.dataset, EDMDataset))
        features_d = dl.dataset.num_atom_types + 1
    
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    model = VarianceDiffusion(egnn_config=EGNNConfig(), num_steps=args.num_steps, schedule=args.noise_schedule, device=args.device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, threshold=args.scheduler_threshold, threshold_mode="rel", min_lr=args.scheduler_min_lr)
    
    if args.checkpoint is not None:
        print(f"Loading model checkpoints using 'model.pth', 'optim.pth' located in {args.checkpoint}")
        # TODO: fix
        # model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))
        # if args.restore_optim_state:
        #     optim.load_state_dict(torch.load(path.join(args.checkpoint, "optim.pth"), map_location=args.device))
        # if args.restore_scheduler_state:
        #     scheduler.load_state_dict(torch.load(path.join(args.checkpoint, "scheduler.pth"), map_location=args.device))        
        # if args.force_start_lr is not None:        
        #     for param_group in optim.param_groups:
        #         param_group['lr'] = args.force_start_lr
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        makedirs(path.join(args.out_dir, args.run_name), exist_ok=True)
        
        train_loss, train_dist, train_grad_norm = 0, 0, 0
        
        if not no_train:
            model.train()
            optim.zero_grad()
            train_loss, train_dist, train_grad_norm = one_epoch(args, epoch, "train", model, gradnorm_queue, dataloaders, wandb_run)
            optim.step()
            scheduler.step(train_loss, epoch)
        
        model.eval()
        with torch.no_grad():
            valid_vlb, valid_dist, _ = one_epoch(args, epoch, "valid", model, gradnorm_queue, dataloaders, wandb_run)
        
        if not no_train:
            # now save things
            torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, "optim.pth"))
            torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, f"optim_epoch_{epoch}.pth"))
            
            torch.save(scheduler.state_dict(), path.join(args.out_dir, args.run_name, "scheduler.pth"))
            torch.save(scheduler.state_dict(), path.join(args.out_dir, args.run_name, f"scheduler_epoch_{epoch}.pth"))
            
            torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, "model.pth"))
            torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, f"model_epoch_{epoch}.pth"))
            
            with open(path.join(args.out_dir, args.run_name, "args.pkl"), "wb") as f:
                pickle.dump(args, f)
            with open(path.join(args.out_dir, args.run_name, f"args_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(args, f)
            
        if wandb_run is not None:
            wandb_run.log({
                "train_compat_loss": train_loss,
                "train_dist": train_dist,
                "train_grad_norm": train_grad_norm,
                "valid_vlb": valid_vlb,
                "valid_dist": valid_dist,
                "lr": optim.param_groups[0]["lr"],
                "epoch": epoch
            })
            
            
def enter_sample(args: Namespace, dataloaders: dict[str, DataLoader], wandb_run: None|Run):
    for _, dl in dataloaders.items():
        assert(isinstance(dl.dataset, EDMDataset))
        features_d = dl.dataset.num_atom_types + 1
    

    model = VarianceDiffusion(egnn_config=EGNNConfig(
        features_d=features_d,
        node_attr_d=0,
        edge_attr_d=0,
        hidden_d=args.hidden_d,
        num_layers=args.num_layers,
        use_tanh=args.use_tanh,
        tanh_range=args.tanh_range,
        use_resid=args.use_resid,
    ), num_steps=args.num_steps, schedule=args.noise_schedule, device=args.device)

    model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=args.device))
