from argparse import Namespace
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from data import EDMDataset, EDMDataloaderItem
from model import EGNNConfig

from .variance import VarianceDiffusion

def one_epoch(args: Namespace, epoch: int, split: Literal["train", "valid", "test"], model: VarianceDiffusion, dataloaders: dict[str, DataLoader]):
    for idx, data in enumerate(tqdm(dataloaders[split], leave=False, unit="batch")):
        data: EDMDataloaderItem
        
        data.to_(args.device)        
        time = torch.randint(low=0, high=args.num_steps + 1, size=(data.coords.shape[0], 1), dtype=torch.long, device=args.device)
        time_float = time.to(torch.float32)
        
        eps_coords = data.demean @ torch.randn_like(data.coords)
        eps_features = torch.randn_like(data.features)
        
        alpha = model.schedule["alpha"][time]
        sigma = model.schedule["sigma"][time]
        
        z_coords = alpha * data.coords + sigma * eps_coords
        z_features = alpha * data.features + sigma * eps_features
        
        pred_eps_coords, pred_eps_features = model.egnn(n_nodes=data.n_nodes,
                                                        coords=data.coords,
                                                        features=data.features,
                                                        edges=data.edges,
                                                        reduce=data.reduce,
                                                        demean=data.demean,
                                                        time=time_float)
        
        pass
    

def enter_train_valid_test_loop(args: Namespace, dataloaders: dict[str, DataLoader]):
    
    for _, dl in dataloaders.items():
        assert(isinstance(dl.dataset, EDMDataset))
        features_d = dl.dataset.num_atom_classes
    
    model = VarianceDiffusion(egnn_config=EGNNConfig(
        features_d=features_d,
        node_attr_d=0,
        edge_attr_d=0,
        hidden_d=args.hidden_d,
        num_layers=args.num_layers,
    ), num_steps=args.num_steps, device=args.device)
    
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), leave=False, unit="epoch"):
        one_epoch(args, epoch, "train", model, dataloaders)