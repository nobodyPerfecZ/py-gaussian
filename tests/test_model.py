import unittest

import numpy as np

from PyGaussian.kernel import RBFKernel
from PyGaussian.model import GaussianProcess


class TestGaussianProcess(unittest.TestCase):
    """
    Tests the class GaussianProcess.
    """

    def setUp(self):
        self.model1 = GaussianProcess(kernel_method="rbf", n_restarts=10)

        self.X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        self.X_test = np.array([
            [4, 5, 6],
        ])

        self.Y = np.array([
            [0.2],
            [0.3],
            [0.4],
        ])

    def test_fit(self):
        """
        Tests the method fit().
        """
        self.model1.fit(self.X, self.Y)

        self.assertIsNotNone(self.model1._thetas)
        self.assertIsNotNone(self.model1._kernel)

    def test_predict(self):
        """
        Tests the method predict().
        """
        self.model1.fit(self.X, self.Y)
        mean = self.model1.predict(self.X_test)

        self.assertTrue(np.isclose(self.Y[1], mean))

    def test_get_kernel_class(self):
        """
        Tests the method _get_kernel_class().
        """
        self.assertEqual(RBFKernel, self.model1._get_kernel_class())

    def test_get_kernel_hps(self):
        """
        Tests the method _get_kernel_hps().
        """
        self.assertEqual({"length_scale": (1e-5, 1e5)}, self.model1._get_kernel_hps())

    def test_get_kernel(self):
        """
        Tests the method _get_kernel().
        """
        self.assertIsInstance(self.model1._get_kernel([0.2]), RBFKernel)


if __name__ == '__main__':
    unittest.main()
