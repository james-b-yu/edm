from argparse import Namespace
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from data import EDMDataset, EDMDataloaderItem
from model import EGNNConfig
from utils.diffusion import Queue, gradient_clipping, scale_inputs

from .variance import VarianceDiffusion

def one_epoch(args: Namespace, epoch: int, split: Literal["train", "valid", "test"], model: VarianceDiffusion, gradnorm_queue: Queue, dataloaders: dict[str, DataLoader]):
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
        
        # # For train likelihood, treat L0 like L12,3,4, etc.
        # dim_coord = (data.n_nodes - 1.) * 3  # dimension of coordinate subspace for each molecule
        
        # loss_coord = 0.5 * torch.sum(((pred_eps_coords - eps_coords) ** 2 / gamma_x[:, None]).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
        #            + 0.5 * torch.sum(dim_coord * (gamma_x_batch.log() - bet_b_batch.log() + bet_b_batch / gamma_x_batch))
        # loss_features = 0.5 * torch.sum(((pred_eps_features - eps_features) ** 2 / gamma_h).sum(axis=-1) * (alf_L_sq * bet_f_sq) / (alf_sq * sig_sq)) \
        #            + 0.5 * torch.sum(data.n_nodes[:, None] * (gamma_h_batch.log() - bet_b_batch[:, None].log() + bet_b_batch[:, None] / gamma_h_batch))
        
        # total_loss = (loss_coord + loss_features)
        
        # if split == "train":
        #     total_loss.backward()
            
            
        ## calculate comparable losses
        # with torch.no_grad():
        loss_coord_compat = (1./3.) * 0.5 * (pred_eps_coords - eps_coords).norm(dim=-1) ** 2
        loss_features_compat = (1./model.egnn.config.features_d) * 0.5 * (pred_eps_features - eps_features).norm(dim=-1) ** 2
        total_loss_compat = data.mean @ (loss_coord_compat + loss_features_compat)
        total_loss_compat = total_loss_compat.mean()
        total_loss_compat.backward()
        
        if args.clip_grad:
            grad_norm = gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.
        
        pbar.set_description(f"Compat loss: {total_loss_compat:.2f} Grad norm: {grad_norm:.2f}")
        
        # pbar.set_description(f"Compat loss: total: {total_loss_compat:.2f}, coord: {loss_coord_compat:.2f}, features: {loss_features_compat:.2f}; Actual: {total_loss:.2f}/{loss_coord:.2f}/{loss_features:.2f}")
        pass
    

def enter_train_valid_test_loop(args: Namespace, dataloaders: dict[str, DataLoader]):
    
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
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        model.train()
        optim.zero_grad()
        one_epoch(args, epoch, "train", model, gradnorm_queue, dataloaders)
        optim.step()
        