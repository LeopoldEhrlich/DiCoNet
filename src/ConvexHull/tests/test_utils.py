import unittest
import numpy as np
import torch

from utils import (
    compute_inclusion_acc,
    compute_exclusion_acc,
    compute_order_acc,
    compute_inclusion_exclusion_acc,
    compute_miss_rate
)

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.long)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # length: number of elements in the input (the universe is [1, ..., length])
        self.lengths = [3, 4, 6, 3, 4, 6, 5]

        # Target: subset-permutation over [1, ..., lengths[i]], then 0s for exclusion, then 0s for padding
        self.target = to_tensor([
            [2, 3, 1, 0, 0, 0, 0, 0],    # length 3, include [2,3,1], no excluded
            [4, 1, 0, 0, 0, 0, 0, 0],    # length 4, include [4,1], exclude [2,3]
            [6, 2, 1, 0, 0, 0, 0, 0],    # length 6, include [6,2,1], exclude [3,4,5]
            [1, 3, 2, 0, 0, 0, 0, 0],    # length 3, include all
            [0, 0, 0, 0, 0, 0, 0, 0],    # length 4, exclude all
            [2, 5, 3, 6, 1, 4, 0, 0],    # length 6, include all
            [3, 5, 4, 2, 1, 0, 0, 0],    # length 5, include all
        ])

        

        # Output: leading 0, then values in [1, ..., lengths[i]], then trailing 0s
        self.output = to_tensor([
            [0, 2, 3, 1, 0, 0, 0, 0, 0],  # match
            [0, 4, 2, 0, 0, 0, 0, 0, 0],  # includes 2 (wrong), misses 1
            [0, 6, 2, 0, 0, 0, 0, 0, 0],  # partial inclusion
            [0, 3, 2, 1, 0, 0, 0, 0, 0],  # correct but shuffled
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # excluded all
            [0, 2, 5, 3, 6, 1, 4, 0, 0],  # perfect match
            [0, 3, 5, 4, 1, 2, 0, 0, 0],  # shuffled
        ])





    def test_inclusion_accuracy(self):
        acc = compute_inclusion_acc(self.output, self.target, self.lengths)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_exclusion_accuracy(self):
        acc = compute_exclusion_acc(self.output, self.target, self.lengths)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_order_accuracy(self):
        acc = compute_order_acc(self.output, self.target, self.lengths)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_inclusion_exclusion_accuracy(self):
        acc = compute_inclusion_exclusion_acc(self.output, self.target, self.lengths)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_miss_rate(self):
        miss = compute_miss_rate(self.output, self.target, self.lengths)
        self.assertAlmostEqual(miss, np.mean([1, 0.75, 5/6, 0, 1, 1, 3/5]))

    def test_only_inclusion(self):
        target = to_tensor([[1, 2, 3, 0]])  # all included but 0 excluded
        output = to_tensor([[0, 1, 2, 0, 0]])  # missing 3
        lengths = [4]
        self.assertLess(compute_inclusion_acc(output, target, lengths), 1.0)

    def test_only_exclusion(self):
        target = to_tensor([[0, 0, 0, 0]])
        output = to_tensor([[0, 1, 2, 3, 0]])
        lengths = [4]

        self.assertEqual(compute_inclusion_acc(output, target, lengths), 0.0)
        self.assertEqual(compute_exclusion_acc(output, target, lengths), 0.25)
        self.assertEqual(compute_inclusion_exclusion_acc(output, target, lengths), 0.25)
        self.assertEqual(compute_miss_rate(output, target, lengths), 0.25)

    def test_full_inclusion_perfect(self):
        target = to_tensor([[1, 2, 3, 4]])
        output = to_tensor([[0, 1, 2, 3, 4]])
        lengths = [4]
        self.assertAlmostEqual(compute_inclusion_acc(output, target, lengths), 1.0)
        self.assertAlmostEqual(compute_order_acc(output, target, lengths), 1.0)

    def test_full_exclusion_perfect(self):
        target = to_tensor([[0, 0, 0, 0]])
        output = to_tensor([[0, 0, 0, 0, 0]])
        lengths = [4]
        self.assertAlmostEqual(compute_exclusion_acc(output, target, lengths), 1.0)

if __name__ == '__main__':
    unittest.main()
