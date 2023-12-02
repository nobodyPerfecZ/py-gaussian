import unittest
import numpy as np

from PyGaussian.kernel import (
    LinearKernel,
    PolynomialKernel,
    SigmoidKernel,
    LaplacianKernel,
    PeriodicKernel,
    RBFKernel,
)


class TestLinearKernel(unittest.TestCase):
    """
    Tests the class LinearKernel.
    """

    def setUp(self):
        self.kernel = LinearKernel()
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 6  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {}

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        self.assertTrue(np.isclose(self.K, self.kernel(self.X1, self.X2), atol=1e-3))

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual(self.hp_types, LinearKernel.get_hps())


class TestPolynomialKernel(unittest.TestCase):
    """
    Tests the class PolynomialKernel.
    """

    def setUp(self):
        self.kernel = PolynomialKernel(bias=1.0, polynomial=0.5)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = np.sqrt(7.0)  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "bias": float,
            "polynomial": float,
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


class TestSigmoidKernel(unittest.TestCase):
    """
    Tests the class SigmoidKernel.
    """

    def setUp(self):
        self.kernel = SigmoidKernel(alpha=1.0, bias=1.0)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 1.0  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "alpha": float,
            "bias": float,
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
        self.assertEqual(self.hp_types, SigmoidKernel.get_hps())


class TestLaplacianKernel(unittest.TestCase):
    """
    Tests the class LaplacianKernel.
    """

    def setUp(self):
        self.kernel = LaplacianKernel(length_scale=1.0)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 0.049788  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "length_scale": float,
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
        self.assertEqual(self.hp_types, LaplacianKernel.get_hps())


class TestPeriodicKernel(unittest.TestCase):
    """
    Tests the class PeriodicKernel.
    """

    def setUp(self):
        self.kernel = PeriodicKernel(length_scale=1.0)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 1.0  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "length_scale": float,
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
        self.assertEqual(self.hp_types, PeriodicKernel.get_hps())


class TestRBFKernel(unittest.TestCase):
    """
    Tests the class RBFKernel.
    """

    def setUp(self):
        self.kernel = RBFKernel(length_scale=1.0)
        self.X1 = np.array([1, 1, 1])
        self.X2 = np.array([2, 2, 2])
        self.K = 0.223  # result of K([1, 1, 1], [2, 2, 2])= ...
        self.hp_types = {
            "length_scale": float,
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
        self.assertEqual(self.hp_types, RBFKernel.get_hps())


if __name__ == '__main__':
    unittest.main()
