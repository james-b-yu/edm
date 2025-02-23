import pytest
from dataclasses import replace
import torch
from data import EDMDataloaderItem, get_dummy_dataloader
from model import EGNN, EGNNConfig

@pytest.fixture
def dummy_dl_cpu():
    """get dummy dataloader on cpu

    Returns:
        Dataloader:
    """
    return get_dummy_dataloader(features_d=7, len=2, max_nodes=25, batch_size=64, device="cpu")

@pytest.fixture
def dummy_dl_cuda():
    """get dummy dataloader on cuda

    Returns:
        Dataloader:
    """
    return get_dummy_dataloader(features_d=7, len=2, max_nodes=25, batch_size=64, device="cuda")

@pytest.fixture
def default_egnn_cpu():
    """get an equivariant graph neural network model on cpu

    Returns:
        EGNN:
    """
    return EGNN(EGNNConfig(num_layers=9, features_d=7, node_attr_d=0, edge_attr_d=0, hidden_d=256))

@pytest.fixture
def default_egnn_cuda():
    """get an equivariant graph neural network model on cuda

    Returns:
        EGNN:
    """
    return EGNN(EGNNConfig(num_layers=9, features_d=7, node_attr_d=0, edge_attr_d=0, hidden_d=256)).cuda()

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

def test_equivariance_of_egnn_cpu(dummy_dl_cpu, default_egnn_cpu):
    """test the equivariance of the EGNN on cpu
    """
    for i, data in enumerate(dummy_dl_cpu):    
        data: EDMDataloaderItem
        Q = _get_random_q(device="cpu")
        x, h = default_egnn_cpu(data)
        
        data_rot = replace(data, coords=data.coords @ Q)
        
        x_rot, h_rot = default_egnn_cpu(data_rot)
        
        assert torch.isclose(x @ Q, x_rot, atol=1e-7, rtol=1e-5).all().item()
        assert torch.isclose(h, h_rot, atol=1e-7, rtol=1e-5).all().item()
        
def test_equivariance_of_egnn_cuda(dummy_dl_cuda, default_egnn_cuda):
    """test the equivariance of the EGNN on cuda
    """
    for i, data in enumerate(dummy_dl_cuda):    
        data: EDMDataloaderItem
        Q = _get_random_q(device="cuda")
        x, h = default_egnn_cuda(data)
        
        data_rot = replace(data, coords=data.coords @ Q)
        
        x_rot, h_rot = default_egnn_cuda(data_rot)
        
        assert torch.isclose(x @ Q, x_rot, atol=1e-7, rtol=1e-5).all().item()
        assert torch.isclose(h, h_rot, atol=1e-7, rtol=1e-5).all().item()