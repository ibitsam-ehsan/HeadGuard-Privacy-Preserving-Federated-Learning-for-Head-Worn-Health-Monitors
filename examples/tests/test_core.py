import unittest
import torch
from headguard import HeadGuard

class TestHeadGuard(unittest.TestCase):
    def setUp(self):
        self.hg = HeadGuard(epsilon=4.0, delta=1e-5)
        self.gradient = torch.randn(100)
    
    def test_init(self):
        self.assertEqual(self.hg.epsilon, 4.0)
        self.assertEqual(self.hg.delta, 1e-5)
        self.assertEqual(self.hg.C, 1.0)
    
    def test_compute_weights(self):
        data = torch.randn(1000)
        weights = self.hg.compute_weights(data)
        self.assertIsNotNone(weights)
        self.assertTrue(len(weights) > 0)
    
    def test_add_noise(self):
        data = torch.randn(1000)
        self.hg.compute_weights(data)
        noisy = self.hg.add_noise(self.gradient)
        self.assertEqual(noisy.shape, self.gradient.shape)

if __name__ == '__main__':
    unittest.main()
