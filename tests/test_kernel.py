import unittest
import numpy as np

from PyGaussian.kernel import PolynomialKernel, GaussianKernel


class TestPolynomialKernel(unittest.TestCase):
    """
    Tests the class PolynomialKernel.
    """

    def setUp(self):
        self.kernel = PolynomialKernel(sigma=0.0, p=2)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 36.0  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "sigma": float,
            "p": float,
        }

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        self.assertTrue(np.isclose(self.K, self.kernel(self.X1, self.X2), atol=1e-3))

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual(self.hp_types, PolynomialKernel.get_hps())


class TestGaussianKernel(unittest.TestCase):
    """
    Tests the class GaussianKernel.
    """

    def setUp(self):
        self.kernel = GaussianKernel(length=1.0)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 0.223  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "length": float,
        }

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        self.assertTrue(np.isclose(self.K, self.kernel(self.X1, self.X2), atol=1e-3))

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual(self.hp_types, GaussianKernel.get_hps())


if __name__ == '__main__':
    unittest.main()
