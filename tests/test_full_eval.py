# to run use below once cd into emd folder
# PYTHONPATH=$(pwd) pytest tests/test_full_eval.py  
# work in progess some don't pass but maybe because they are not quite demanding the right output 

import sys
sys.argv = [""]

import torch
import pytest

# Import the functions to be tested
from extensions.vanilla.full_eval import compute_atom_stability, compute_molecule_stability, compute_nll


# Test data for atoms (one-hot encoding for H, C, N, O, F)
one_hot_example = torch.tensor([
    [1, 0, 0, 0, 0],  # Hydrogen
    [0, 1, 0, 0, 0],  # Carbon
    [0, 0, 1, 0, 0],  # Nitrogen
    [0, 0, 0, 1, 0],  # Oxygen
    [0, 0, 0, 0, 1],  # Fluorine
], dtype=torch.float32)

charges_example = torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.float32)
node_mask_example = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)

@pytest.mark.parametrize("one_hot, charges, node_mask, expected", [
    # All atoms are stable (H=1, C=4, N=3, O=2, F=1 with charge 0)
    (one_hot_example, charges_example, node_mask_example, torch.tensor([True, True, True, True, True])),
    
    # Introduce unstable atom (Oxygen with incorrect charge)
    (one_hot_example, torch.tensor([[0], [0], [0], [2], [0]], dtype=torch.float32), node_mask_example,
    torch.tensor([True, True, True, False, True])),

    # Only Carbon, which should always be stable at valency 4
    (torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32), torch.tensor([[0]], dtype=torch.float32),
    torch.tensor([1], dtype=torch.float32), torch.tensor([True])),
])
def test_compute_atom_stability(one_hot, charges, node_mask, expected):
    result = compute_atom_stability(one_hot, charges, node_mask)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

    """
    This function defines parameterized test cases for testing the `compute_molecule_stability` function
    with different input scenarios.
    
    :param one_hot: The `one_hot` parameter in the `@pytest.mark.parametrize` decorator represents a
    one-hot encoded representation of the molecules in your test cases. It is used to provide input data
    to the `test_compute_molecule_stability` test function for testing the `compute_molecule_stability`
    function
    :param charges: The `charges` parameter in the `@pytest.mark.parametrize` decorator represents the
    charges of atoms in a molecule. It is used to test the `compute_molecule_stability` function with
    different scenarios where the stability of molecules is determined based on the charges of atoms
    :param n_nodes: The `n_nodes` parameter in the `@pytest.mark.parametrize` decorator represents the
    number of nodes in the molecular graph. In the context of your test cases, it is a tensor containing
    the number of nodes for each molecule
    :param expected_stability: The `expected_stability` parameter in the `@pytest.mark.parametrize`
    decorator represents the expected stability value that the `compute_molecule_stability` function
    should return for a given set of input parameters (`one_hot`, `charges`, `n_nodes`)
    """
@pytest.mark.parametrize("one_hot, charges, n_nodes, expected_stability", [
    # Single stable molecule (all atoms stable)
    (one_hot_example, charges_example, torch.tensor([5]), 100.0),

    # Some unstable atoms, molecule becomes unstable
    (one_hot_example, torch.tensor([[0], [0], [0], [2], [0]], dtype=torch.float32), torch.tensor([5]), 0.0),

    # Multiple molecules, some stable, some not
    (torch.cat([one_hot_example, one_hot_example]), torch.cat([charges_example, charges_example]),
    torch.tensor([5, 5]), 100.0),

    # Mixed stability, one unstable molecule
    (torch.cat([one_hot_example, one_hot_example]), torch.cat([charges_example,
    torch.tensor([[0], [0], [0], [2], [0]], dtype=torch.float32)]), torch.tensor([5, 5]), 50.0),
])
def test_compute_molecule_stability(one_hot, charges, n_nodes, expected_stability):
    result = compute_molecule_stability(one_hot, charges, n_nodes)
    assert abs(result - expected_stability) < 1e-5, f"Expected {expected_stability}, but got {result}"

@pytest.mark.parametrize("predictions, targets, expected_nll", [
    # Simple case: single correct prediction
    (torch.tensor([[0.1, 0.9]]), torch.tensor([1]), -torch.log(torch.tensor(0.9)).item()),

    # Incorrect prediction
    (torch.tensor([[0.9, 0.1]]), torch.tensor([1]), -torch.log(torch.tensor(0.1)).item()),

    # Multiple predictions
    (torch.tensor([[0.7, 0.3], [0.2, 0.8]]), torch.tensor([0, 1]),
    (-torch.log(torch.tensor(0.7)) - torch.log(torch.tensor(0.8))).item() / 2),
])
def test_compute_nll(predictions, targets, expected_nll):
    """
    The function `test_compute_nll` is a unit test that checks if the computed negative log-likelihood
    (NLL) matches the expected NLL within a small margin of error.
    
    :param predictions: It seems like you were about to provide more information about the `predictions`
    parameter for the `test_compute_nll` function, but the message got cut off. Could you please provide
    more details or complete the information so that I can assist you further?
    :param targets: Targets typically refer to the true values or labels in a machine learning context.
    They are the values that the model is trying to predict or classify. In the context of the
    `test_compute_nll` function, the targets parameter likely represents the true labels for the
    predictions made by the model
    :param expected_nll: The `expected_nll` parameter in the `test_compute_nll` function represents the
    expected negative log likelihood (NLL) value that the `compute_nll` function should return when
    given the `predictions` and `targets` as inputs. The test function compares the computed NLL value
    with
    """
    result = compute_nll(predictions, targets)
    assert abs(result - expected_nll) < 1e-5, f"Expected {expected_nll}, but got {result}"
