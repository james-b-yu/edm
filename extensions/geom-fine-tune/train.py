import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import random
import numpy as np

# Import your dataset classes
from qm9_dataset import QM9Dataset  # Your QM9 dataset class
from geom_dataset import GEOMDataset  # Your GEOM dataset class

# Load both datasets
qm9_dataset = QM9Dataset(root="data/qm9")  # Small molecules
geom_dataset = GEOMDataset(root="data/geom", max_atoms=40)  # Larger molecules (subset)

# Define batch size
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

# Split into training & validation
qm9_train_size = int(TRAIN_SPLIT * len(qm9_dataset))
qm9_val_size = len(qm9_dataset) - qm9_train_size
geom_train_size = int(TRAIN_SPLIT * len(geom_dataset))
geom_val_size = len(geom_dataset) - geom_train_size

qm9_train, qm9_val = random_split(qm9_dataset, [qm9_train_size, qm9_val_size])
geom_train, geom_val = random_split(geom_dataset, [geom_train_size, geom_val_size])

# Create dataloaders
qm9_train_loader = DataLoader(qm9_train, batch_size=BATCH_SIZE, shuffle=True)
geom_train_loader = DataLoader(geom_train, batch_size=BATCH_SIZE, shuffle=True)
qm9_val_loader = DataLoader(qm9_val, batch_size=BATCH_SIZE, shuffle=False)
geom_val_loader = DataLoader(geom_val, batch_size=BATCH_SIZE, shuffle=False)

# Function to fetch mixed batches
def get_mixed_batch():
    """
    Returns a batch with 50% QM9 molecules and 50% GEOM molecules.
    """
    qm9_batch = next(iter(qm9_train_loader))  # Sample from QM9
    geom_batch = next(iter(geom_train_loader))  # Sample from GEOM

    # Combine batches
    return {
        "one_hot": torch.cat([qm9_batch["one_hot"], geom_batch["one_hot"]], dim=0),
        "charges": torch.cat([qm9_batch["charges"], geom_batch["charges"]], dim=0),
        "positions": torch.cat([qm9_batch["positions"], geom_batch["positions"]], dim=0),
        "node_mask": torch.cat([qm9_batch["node_mask"], geom_batch["node_mask"]], dim=0),
        "edge_mask": torch.cat([qm9_batch["edge_mask"], geom_batch["edge_mask"]], dim=0),
        "batch_size": qm9_batch["batch_size"] + geom_batch["batch_size"],
    }