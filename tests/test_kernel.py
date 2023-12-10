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
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [12, 15, 18],
            [24, 30, 36],
            [36, 45, 54],
        ]), K)
        np.testing.assert_allclose(np.array([
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(12, k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose(0, k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({}, LinearKernel.get_hps())


class TestPolynomialKernel(unittest.TestCase):
    """
    Tests the class PolynomialKernel.
    """

    def setUp(self):
        self.kernel = PolynomialKernel(bias=1.0, polynomial=0.5)
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [3.60555128, 4., 4.35889894],
            [5., 5.56776436, 6.08276253],
            [6.08276253, 6.78232998, 7.41619849],
        ]), K)
        np.testing.assert_allclose(np.array([
            [
                [0.13867505, 9.24805643],
                [0.125, 11.09035489],
                [0.11470787, 12.83451196],
            ], [
                [0.1, 16.09437912],
                [0.08980265, 19.11963158],
                [0.08219949, 21.96435618],
            ], [
                [0.08219949, 21.96435618],
                [0.07372098, 25.96710934],
                [0.06741999, 29.71917831],
            ],
        ]
        ), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(3.60555128, k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose([0.13867505, 9.24805643], k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({
            "bias": (1e-5, 1e5),
            "polynomial": (1e-5, 1e5),
        }, PolynomialKernel.get_hps())


class TestSigmoidKernel(unittest.TestCase):
    """
    Tests the class SigmoidKernel.
    """

    def setUp(self):
        self.kernel = SigmoidKernel(alpha=1.0, bias=1.0)
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]), K)
        np.testing.assert_allclose(np.array([
            [
                [2.45236273e-10, 2.04363561e-11],
                [7.59849933e-13, 5.06566622e-14],
                [2.26017561e-15, 1.25565312e-16],
            ], [
                [1.85159985e-20, 7.71499939e-22],
                [1.42207784e-25, 4.74025946e-27],
                [1.04850579e-30, 2.91251607e-32],
            ], [
                [1.04850579e-30, 2.91251607e-32],
                [1.99609023e-38, 4.43575608e-40],
                [3.64804966e-46, 6.75564752e-48],
            ],
        ]
        ), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(1., k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose([2.45236273e-10, 2.04363561e-11], k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({
            "alpha": (1e-5, 1e5),
            "bias": (1e-5, 1e5),
        }, SigmoidKernel.get_hps())


class TestLaplacianKernel(unittest.TestCase):
    """
    Tests the class LaplacianKernel.
    """

    def setUp(self):
        self.kernel = LaplacianKernel(length_scale=1.0)
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [1.23409804e-04, 6.14421235e-06, 3.05902321e-07],
            [2.47875218e-03, 1.23409804e-04, 6.14421235e-06],
            [4.97870684e-02, 2.47875218e-03, 1.23409804e-04],
        ]), K)
        np.testing.assert_allclose(np.array([
            [[1.11068824e-03], [7.37305482e-05], [4.58853481e-06]],
            [[1.48725131e-02], [1.11068824e-03], [7.37305482e-05]],
            [[1.49361205e-01], [1.48725131e-02], [1.11068824e-03]],
        ]), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(1.23409804e-04, k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose(1.11068824e-03, k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({
            "length_scale": (1e-5, 1e5),
        }, LaplacianKernel.get_hps())


class TestPeriodicKernel(unittest.TestCase):
    """
    Tests the class PeriodicKernel.
    """

    def setUp(self):
        self.kernel = PeriodicKernel(period=1.0, length_scale=1.0)
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]), K)
        np.testing.assert_allclose(np.array([
            [
                [1.24653861e-13, 4.85922170e-30],
                [2.21606864e-13, 8.63861635e-30],
                [1.01593148e-12, 1.16194769e-28],
            ], [
                [5.54017160e-14, 2.15965409e-30],
                [1.24653861e-13, 4.85922170e-30],
                [2.21606864e-13, 8.63861635e-30],
            ], [
                [1.38504290e-14, 5.39913522e-31],
                [5.54017160e-14, 2.15965409e-30],
                [1.24653861e-13, 4.85922170e-30],
            ],
        ]
        ), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(1., k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose([1.24653861e-13, 4.85922170e-30], k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({
            "period": (1e-5, 1e5),
            "length_scale": (1e-5, 1e5),
        }, PeriodicKernel.get_hps())


class TestRBFKernel(unittest.TestCase):
    """
    Tests the class RBFKernel.
    """

    def setUp(self):
        self.kernel = RBFKernel(length_scale=1.0)
        self.X1 = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        self.X2 = np.array([
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        K, K_gradient = self.kernel(self.X1, self.X2, return_gradient=True)
        np.testing.assert_allclose(np.array([
            [1.37095909e-06, 3.77513454e-11, 5.17555501e-17],
            [2.47875218e-03, 1.37095909e-06, 3.77513454e-11],
            [2.23130160e-01, 2.47875218e-03, 1.37095909e-06],
        ]), K)
        np.testing.assert_allclose(np.array([
            [[3.70158953e-05], [1.81206458e-09], [3.88166625e-15]],
            [[2.97450261e-02], [3.70158953e-05], [1.81206458e-09]],
            [[6.69390480e-01], [2.97450261e-02], [3.70158953e-05]],
        ]), K_gradient)

    def test_single_call(self):
        """
        Tests the method _call().
        """
        k = self.kernel._call(self.X1[0], self.X2[0])
        np.testing.assert_allclose(1.37095909e-06, k)

    def test_gradient(self):
        """
        Tests the method _gradient().
        """
        k_gradient = self.kernel._gradient(self.X1[0], self.X2[0])
        np.testing.assert_allclose(3.70158953e-05, k_gradient)

    def test_get_hps(self):
        """
        Tests the staticmethod get_hps().
        """
        self.assertEqual({
            "length_scale": (1e-5, 1e5),
        }, RBFKernel.get_hps())


if __name__ == '__main__':
    unittest.main()
