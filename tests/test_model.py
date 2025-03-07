import sys
sys.argv = [""]

import pytest
from dataclasses import replace
import torch
from data import EDMDataloaderItem, get_dummy_dataloader, get_qm9_dataloader
from model import EGCL, EGNN, EGCLConfig, EGNNConfig
from tqdm import tqdm

@pytest.fixture
def dummy_dl():
    """get dummy dataloader on cpu

    Returns:
        Dataloader:
    """
    return get_dummy_dataloader(num_atom_classes=5, len=1000, max_nodes=25, batch_size=64)

@pytest.fixture
def qm9_h_train_dl():
    """get dummy dataloader on cuda

    Returns:
        Dataloader:
    """
    return get_qm9_dataloader(use_h=True, split="train", batch_size=64)

@pytest.fixture
def default_egcl_cpu():
    """get an equivariant graph covolutional layer module on cpu

    Returns:
        EGCL:
    """
    return EGCL(EGCLConfig(features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256))

@pytest.fixture
def default_egcl_cuda():
    """get an equivariant graph covolutional layer module on cuda

    Returns:
        EGCL:
    """
    return EGCL(EGCLConfig(features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256)).cuda()

@pytest.fixture
def default_egnn_cpu():
    """get an equivariant graph neural network module on cpu

    Returns:
        EGNN:
    """
    return EGNN(EGNNConfig(num_layers=9, features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256))

@pytest.fixture
def default_egnn_cuda():
    """get an equivariant graph neural network module on cuda

    Returns:
        EGNN:
    """
    return EGNN(EGNNConfig(num_layers=9, features_d=5, node_attr_d=0, edge_attr_d=0, hidden_d=256)).cuda()

def _get_random_q(dim=3, device: torch.device|str="cpu"):
    """get a random orthonormal [dim, dim] matrix

    Args:
        dim (int, optional): size of matrix. Defaults to 3.
        device (torch.device | str, optional): device. Defaults to "cpu"

    Returns:
        torch.Tensor:
    """
    Q, _ = torch.linalg.qr(torch.randn(size=(3, 3), device=device))
    return Q

def test_equivariance_of_egcl_cpu(dummy_dl, default_egcl_cpu):
    """Test the equivariance of the EGCL on CPU
    
    This loads an epoch of dummy coordinate and feature data and feeds it into the EGCL. Within each minibatch, we rotate coordinates by a random orthonormal matrix and check that a) the predictions for coordinate residual are equivariant, and b) the predictions for feature residual are invariant
    """
    for i, data in enumerate(tqdm(dummy_dl, desc="Testing equivariance of a single EGCL on CPU")):
        data: EDMDataloaderItem
        Q = _get_random_q(device="cpu")
        
        x, h = default_egcl_cpu(coords=data.coords, features=data.one_hot, edges=data.edges, reduce=data.reduce)
        x_rot, h_rot = default_egcl_cpu(coords=data.coords @ Q, features=data.one_hot, edges=data.edges, reduce=data.reduce)

        assert torch.isclose(x @ Q, x_rot, atol=1e-5, rtol=1e-5).all()
        assert torch.isclose(h, h_rot, atol=1e-5, rtol=1e-5).all()

def test_equivariance_of_egcl_cuda(dummy_dl, default_egcl_cuda):
    """Test the equivariance of the EGCL on CUDA
    See test_equivariance_of_egcl_cpu for notes
    """
    for i, data in enumerate(tqdm(dummy_dl, desc="Testing equivariance of a single EGCL on CUDA")):
        data: EDMDataloaderItem
        Q = _get_random_q(device="cuda")
        data.to_(device="cuda")

        x, h = default_egcl_cuda(coords=data.coords, features=data.one_hot, edges=data.edges, reduce=data.reduce)
        x_rot, h_rot = default_egcl_cuda(coords=data.coords @ Q, features=data.one_hot, edges=data.edges, reduce=data.reduce)

        assert torch.isclose(x @ Q, x_rot, atol=1e-5, rtol=1e-5).all()
        assert torch.isclose(h, h_rot, atol=1e-5, rtol=1e-5).all()
        
        
def test_equivariance_of_egnn_cpu(dummy_dl, default_egnn_cpu):
    """Test the equivariance of the EGNN on CPU. Due to numerical instabilities, we have a more relaxed relative tolerance than the test for equivariance of EGCL, and we only require at least 99% of tensors to be "close" to pass the test of equivariance
    """
    for i, data in enumerate(tqdm(dummy_dl, desc="Testing equivariance of EGNN network on CPU")):
        data: EDMDataloaderItem
        Q = _get_random_q(device="cpu")
        
        time = float(torch.randint(low=0, high=1001, size=()) / 1000)
        x, h = default_egnn_cpu(data.n_nodes, data.coords, data.one_hot, data.edges, data.reduce, data.demean, time=time)
        x_rot, h_rot = default_egnn_cpu(data.n_nodes, data.coords @ Q, data.one_hot, data.edges, data.reduce, data.demean, time=time)

        assert torch.isclose(x @ Q, x_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99
        assert torch.isclose(h, h_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99
        
def test_equivariance_of_egnn_cuda(dummy_dl, default_egnn_cuda):
    """Test the equivariance of the EGNN on CPU. Due to numerical instabilities, we have a more relaxed relative tolerance than the test for equivariance of EGCL, and we only require at least 99% of tensors to be "close" to pass the test of equivariance
    """
    for i, data in enumerate(tqdm(dummy_dl, desc="Testing equivariance of EGNN network on CUDA")):
        data: EDMDataloaderItem
        Q = _get_random_q(device="cuda")
        data.to_("cuda")
        
        time = float(torch.randint(low=0, high=1001, size=()) / 1000)
        x, h = default_egnn_cuda(data.n_nodes, data.coords, data.one_hot, data.edges, data.reduce, data.demean, time=time)
        x_rot, h_rot = default_egnn_cuda(data.n_nodes, data.coords @ Q, data.one_hot, data.edges, data.reduce, data.demean, time=time)

        assert torch.isclose(x @ Q, x_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99
        assert torch.isclose(h, h_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99
        
        
def test_equivariance_of_egnn_cuda_on_qm9(qm9_h_train_dl, default_egnn_cuda):
    """Test the equivariance of the EGNN on CPU by running one epoch of QM9 with hydrogens.
    """
    for i, data in enumerate(tqdm(qm9_h_train_dl, desc="Testing equivariance on CUDA during one QM9 epoch")):
        data: EDMDataloaderItem
        Q = _get_random_q(device="cuda")
        data.to_("cuda")
        
        time = float(torch.randint(low=0, high=1001, size=()) / 1000)
        x, h = default_egnn_cuda(data.n_nodes, data.coords, data.one_hot, data.edges, data.reduce, data.demean, time=time)
        x_rot, h_rot = default_egnn_cuda(data.n_nodes, data.coords @ Q, data.one_hot, data.edges, data.reduce, data.demean, time=time)

        assert torch.isclose(x @ Q, x_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99
        assert torch.isclose(h, h_rot, rtol=1e-3).to(torch.float32).mean() >= 0.99