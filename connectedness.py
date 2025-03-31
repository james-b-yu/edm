import torch
from configs.dataset_reg_config import get_dataset_info
from extensions.regularization.disconnection_penalty import get_adjacency, get_graph_components


def check_connectedness(coords, features, dataset_info):
    num_types = len(dataset_info['atom_types'])

    coords = coords.view(-1, 3)
    features = features[:num_types].view(-1, num_types).type(torch.float32)
    atom_types = torch.argmax(features, dim=1)
    
    A, distances = get_adjacency(coords, atom_types, dataset_info)
    components = get_graph_components(A)
    is_connected = (len(components) == 1)
    
    return is_connected
    
    
def compute_connectedness(samples, args):
    use_h = "_no_h" not in args.dataset
    dataset_name = args.dataset.split('_')[0]
    dataset_info = get_dataset_info(dataset_name, use_h)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    samples_torch = [(torch.from_numpy(s[0]).to(device), torch.from_numpy(s[1]).to(device)) for s in samples]
    
    res = [check_connectedness(s[0], s[1], dataset_info) for s in samples_torch]
    percentage_molecules_connected = torch.mean(res) * 100
    
    logfile = f"sampling_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(logfile, "a+") as f:
        f.write(f"percentage_molecules_connected: {percentage_molecules_connected:.4f} %\n")
    
    return percentage_molecules_connected, res