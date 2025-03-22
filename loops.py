from argparse import Namespace
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from os import path, makedirs
import pickle
from wandb.wandb_run import Run

from data import EDMDataloaderItem
from utils.diffusion import Queue, gradient_clipping

from models.base import BaseEDM

def one_train_epoch(args: Namespace, epoch: int, dl: DataLoader, model: BaseEDM, model_ema: BaseEDM, gradnorm_queue: Queue, optim: torch.optim.Optimizer, wandb_run: None|Run):
    model.train()
    
    losses: list[float] = []
    grad_norms: list[float] = []
    dl_len = len(dl)
    
    for idx, data in enumerate(pbar := tqdm(dl, leave=False, unit="batch")):
        data: EDMDataloaderItem
        data.to_(args.device)

        optim.zero_grad()
    
        train_loss = model.get_mse(data=data)
        assert isinstance(train_loss, torch.Tensor)
        train_loss.backward()
        
        if args.clip_grad:
            grad_norm = gradient_clipping(model, gradnorm_queue, max=args.max_grad_norm)
        
        optim.step()
        
        _update_ema(args.ema_beta, model, model_ema)
        
        pbar.set_description(f"(train) Total loss: {train_loss:.2f} Grad norm: {grad_norm:.2f}")
        
        if wandb_run is not None:
            wandb_run.log({
                "train_batch_loss": train_loss,
                "train_batch_grad_norm": grad_norm,
                "epoch": float(epoch) + idx/dl_len
            })
            
        losses.append(float(train_loss))
        grad_norms.append(float(grad_norm))
        
    return sum(losses) / dl_len, sum(grad_norms) / dl_len

@torch.no_grad()
def one_valid_epoch(args: Namespace, split: Literal["valid", "test"], epoch: int, dl: DataLoader, model: BaseEDM):
    model.eval()
    
    vlbs: list[float]  = []
    dists: list[float] = []
    dl_len = len(dl)
    
    for idx, data in enumerate(pbar := tqdm(dl, leave=False, unit="batch")):
        data: EDMDataloaderItem
        data.to_(args.device)
        
        vlb, dist = model.estimate_vlb(data=data)
        
        pbar.set_description(f"({split}) Vlb: {vlb:.2f} Dist: {dist:.2f}")
        
        vlbs.append(float(vlb))
        dists.append(float(dist))
        
    return sum(vlbs) / dl_len, sum(dists) / dl_len
    

def enter_train_loop(model: BaseEDM, model_ema: BaseEDM, optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, args: Namespace, train_dl: DataLoader, valid_dl: DataLoader, wandb_run: None|Run):
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        # first train
        train_loss, train_grad_norm = one_train_epoch(args, epoch, train_dl, model, model_ema, gradnorm_queue, optim, wandb_run)
        
        # now eval
        with torch.no_grad():
            valid_vlb, valid_dist = one_valid_epoch(args, "valid", epoch, valid_dl, model)
            ema_valid_vlb, ema_valid_dist = one_valid_epoch(args, "valid", epoch, valid_dl, model_ema)
            
        # step the scheduler
        scheduler.step(valid_vlb)  # type: ignore
        
        # now save things
        _save_checkpoint(args, epoch, model, model_ema, optim, scheduler)
            
        # now log things
        if wandb_run is not None:
            wandb_run.log({
                "train_loss": train_loss,
                "train_grad_norm": train_grad_norm,
                "valid_vlb": valid_vlb,
                "valid_dist": valid_dist,
                "ema_valid_vlb": ema_valid_vlb,
                "ema_valid_dist": ema_valid_dist,
                "lr": optim.param_groups[0]["lr"],
                "epoch": epoch
            })

def enter_valid_loop(model: BaseEDM, split: Literal["valid", "test"], args: Namespace, dl: DataLoader):
    """go through valid or test set many times and return point estimate and estimated std of the point estimate of the vlb. Go through the set `args.reruns` times

    Args:
        model (BaseEDM): the model
        split ("valid" | "test"): the name of the split (for printing purposes only)
        args (Namespace): arguments
        dl (DataLoader): dataloader
        
    Returns tuple[float, float]: point estimate and estimated std of this point estimate of the vlbs across runs
    """
    vlbs = []
    for _ in tqdm(range(args.reruns)):
        vlb, _ = one_valid_epoch(args, split, 0, dl, model)
        vlbs.append(vlb)
    vlbs = torch.tensor(vlbs, dtype=torch.float32, device="cpu")
    return float(vlbs.mean()), float(vlbs.std())
  

def _update_ema(beta: float, model: BaseEDM, model_ema: BaseEDM):
    """update ema with new model params via formula model_ema = beta * model_ema + (1 - beta) * model

    Args:
        beta (float): _description_
        model (BaseEDM): _description_
        model_ema (BaseEDM): _description_
    """
    for (current_name, current_param), (ema_name, ema_param) in zip(model.named_parameters(), model_ema.named_parameters()):
        assert current_name == ema_name
        ema_param.data = beta * ema_param.data + (1 - beta) * current_param.data
            
def _save_checkpoint(args: Namespace, epoch: int, model: BaseEDM, model_ema: BaseEDM, optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    """save data for a checkpoint

    Args:
        args (Namespace): _description_
        epoch (int): _description_
        model (BaseEDM): _description_
        model_ema (BaseEDM): _description_
        optim (torch.optim.Optimizer): _description_
        scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
    """
    makedirs(path.join(args.out_dir, args.run_name), exist_ok=True)
    torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, "model.pth"))
    torch.save(model.state_dict(), path.join(args.out_dir, args.run_name, f"model_epoch_{epoch}.pth"))
    
    torch.save(model_ema.state_dict(), path.join(args.out_dir, args.run_name, "model_ema.pth"))
    torch.save(model_ema.state_dict(), path.join(args.out_dir, args.run_name, f"model_ema_epoch_{epoch}.pth"))
    
    torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, "optim.pth"))
    torch.save(optim.state_dict(), path.join(args.out_dir, args.run_name, f"optim_epoch_{epoch}.pth"))
    
    torch.save(scheduler.state_dict(), path.join(args.out_dir, args.run_name, "scheduler.pth"))
    torch.save(scheduler.state_dict(), path.join(args.out_dir, args.run_name, f"scheduler_epoch_{epoch}.pth"))
    
    with open(path.join(args.out_dir, args.run_name, "args.pkl"), "wb") as f:
        pickle.dump(args, f)
    with open(path.join(args.out_dir, args.run_name, f"args_epoch_{epoch}.pkl"), "wb") as f:
        pickle.dump(args, f)