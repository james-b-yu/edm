import unittest
import torch
from qm9 import bond_analyze
from configs.datasets_config import get_dataset_info
from extensions.vanilla.full_eval import compute_atom_stability

class TestComputeAtomStability(unittest.TestCase):

    def setUp(self):
        self.dataset_info = get_dataset_info(remove_h=False)
        self.atom_decoder = self.dataset_info['atom_decoder']

    def test_single_molecule(self):
        one_hot = torch.tensor([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]], dtype=torch.float32)  # C, C
        charges = torch.tensor([[[0], [0]]], dtype=torch.float32)
        coords = torch.tensor([[[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]]], dtype=torch.float32)  # Bond length for C-C
        node_mask = torch.tensor([[1, 1]], dtype=torch.bool)

        stability = compute_atom_stability(one_hot, charges, coords, node_mask, self.dataset_info)
        print(f"stability: {stability}")
        self.assertTrue(stability.all().item(), "Atoms should be stable")

    # def test_no_bond(self):
    #     one_hot = torch.tensor([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]], dtype=torch.float32)  # C, C
    #     charges = torch.tensor([[[0], [0]]], dtype=torch.float32)
    #     coords = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float32)  # No bond length for C-C
    #     node_mask = torch.tensor([[1, 1]], dtype=torch.bool)

    #     stability = compute_atom_stability(one_hot, charges, coords, node_mask, self.dataset_info)
    #     self.assertFalse(stability.any().item(), "Atoms should not be stable")

    # def test_multiple_atoms(self):
    #     one_hot = torch.tensor([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]], dtype=torch.float32)  # C, C, N
    #     charges = torch.tensor([[[0], [0], [0]]], dtype=torch.float32)
    #     coords = torch.tensor([[[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.54, 0.0, 0.0]]], dtype=torch.float32)  # C-C and C-N bond lengths
    #     node_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    #     stability = compute_atom_stability(one_hot, charges, coords, node_mask, self.dataset_info)
    #     self.assertTrue(stability.all().item(), "All atoms should be stable")

    # def test_partial_stability(self):
    #     one_hot = torch.tensor([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]], dtype=torch.float32)  # C, C, N
    #     charges = torch.tensor([[[0], [0], [0]]], dtype=torch.float32)
    #     coords = torch.tensor([[[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=torch.float32)  # Only C-C bond length
    #     node_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    #     stability = compute_atom_stability(one_hot, charges, coords, node_mask, self.dataset_info)
    #     self.assertFalse(stability.all().item(), "Not all atoms should be stable")
    #     self.assertTrue(stability[0, 0].item(), "First atom should be stable")
    #     self.assertTrue(stability[0, 1].item(), "Second atom should be stable")
    #     self.assertFalse(stability[0, 2].item(), "Third atom should not be stable")

if __name__ == '__main__':
    unittest.main()
