from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path

from masked_model import MaskedEDM, get_config_from_args

@torch.no_grad()
def do_demo(args: Namespace, dl: DataLoader):
    print("This is a demo run which calculates mean-squared error on the validation (not test!) dataset")
    config = get_config_from_args(args, dl.dataset.num_atom_types)  # type:ignore
    model = MaskedEDM(config)
    model.load_state_dict(torch.load(path.join(args.checkpoint, "model.pth"), map_location=config.device))
    model.eval()
    
    for idx, data in enumerate(pbar := tqdm(dl)):
        data._to(dtype=torch.float32, device=args.device)
        batch_size = data["batch_size"]  # note this may vary over batch because the final batch of a dataset may be smaller
        time_int = torch.randint(low=0, high=args.num_steps + 1, size=(batch_size, ), device=args.device, dtype=torch.long)
        
        (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = model.get_eps_and_predicted_eps(data["positions"], data["one_hot"], data["charges"], time_int, data["node_mask"], data["edge_mask"])
        
        sq_coord_err = (eps_coord - pred_eps_coord) ** 2
        sq_feat_err  = (eps_feat - pred_eps_feat) ** 2
        
        mse = torch.concat([sq_coord_err, sq_feat_err], dim=-1).mean()
        pbar.set_description(f"Batch MSE {mse:.2f}")