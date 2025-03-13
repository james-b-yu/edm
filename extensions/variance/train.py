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
from model_config import get_config_from_args
from utils.diffusion import Queue, gradient_clipping, scale_features

from .variance_model import VarianceEDM

def one_epoch(args: Namespace, epoch: int, split: Literal["train", "valid", "test"], model: VarianceEDM, gradnorm_queue: Queue, dl: DataLoader, optim: torch.optim.Optimizer|None, wandb_run: None|Run):
    losses: list[float] = []
    dists: list[float] = []
    grad_norms: list[float] = []
    
    vlbs: list[float] = []
    
    for idx, data in enumerate(pbar := tqdm(dl, leave=False, unit="batch")):
        data: EDMDataloaderItem
        
        data.to_(args.device)
        if split == "train":
            assert(optim is not None)
            optim.zero_grad()
            
        compat_loss, dist = model.calculate_loss(args=args, split=split, data=data)        
        grad_norm = 0.
        if split == "train":
            dist.backward()
            if args.clip_grad:
                grad_norm = gradient_clipping(model, gradnorm_queue, max=args.max_grad_norm)
            optim.step()
        
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
    

def enter_train_loop(model: VarianceEDM, optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, args: Namespace, train_dl: DataLoader, valid_dl: DataLoader, wandb_run: None|Run):
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        # first train
        model.train()
        train_loss, train_dist, train_grad_norm = one_epoch(args, epoch, "train", model, gradnorm_queue, train_dl, optim, wandb_run)
        scheduler.step(train_dist)
        
        # now eval
        model.eval()
        with torch.no_grad():
            valid_vlb, valid_dist, _ = one_epoch(args, epoch, "valid", model, gradnorm_queue, valid_dl, None, wandb_run)
        
        # now save things
        makedirs(path.join(args.out_dir, args.run_name), exist_ok=True)
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