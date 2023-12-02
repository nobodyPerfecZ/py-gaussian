from typing import Type
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np

from PyGaussian.kernel import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    SigmoidKernel,
    LaplacianKernel,
    PeriodicKernel,
    RBFKernel,
)


class GaussianProcess:
    """
    Class to represent the stochastic model Gaussian Processes.

    More information about how to implement a Gaussian Process can you find here:
    https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce#9f9c
    """

    def __init__(
            self,
            kernel_method: str = "rbf",
            optimizer: str = "L-BFGS-B",
            n_restarts: int = 20,
            return_cov: bool = False,
    ):
        self._optimizer = optimizer
        self._n_restarts = n_restarts
        self._return_cov = return_cov

        # Kernel function for calculating the covariance function K(X1, X2)
        self._kernel_method = kernel_method
        self._kernel = None
        self._thetas = None  # contains all hyperparameters for the kernel function

        # Dataset, used to train the hyperparameters theta
        self._X = None
        self._Y = None

        # Parameters for .predict() method
        self._K_obv_obv = None
        self._K_obv_obv_inv = None
        self._mean = None
        self._variance = None

    def cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Returns the covariance matrix between X1 and X2, calculated by the covariance function K(X1, X2).

        Args:
            X1 (np.ndarray):
                Dataset of shape (N, M)
            X2 (np.ndarray):
                Dataset of shape (N, M)

        Returns:
            np.ndarray:
                Covariance matrix of shape (M, M)
        """
        return np.array([[self._kernel(x1, x2) for x2 in X2] for x1 in X1])

    def negative_likelihood(self, thetas: np.ndarray) -> float:
        """
        Objective function of the Gaussian Process to search for good hyperparameters of the kernel function
        (negative likelihood).

        Maximization Problem:
        thetas = argmax[n/2 ln(std^2) - 1/2 ln(det(K))]

        Minimization Problem
        thetas = argmin[-(n/2 ln(std^2) - 1/2 ln(det(K)))]

        Args:
            thetas (np.ndarray): single sample of possible thetas (hyperparameters) for the kernel function

        Returns:
            float: negative likelihood value
        """
        n = len(self._X)  # number of training instances
        one = np.ones((len(self._X), 1))  # vector of ones

        # Construct the kernel function K(X1, X2) with given thetas
        self._kernel = self._get_kernel(*thetas)

        # Construct covariance matrix K_obv_obv
        K_obv_obv = self.cov(self._X, self._X) + np.identity(n) * 1e-6

        # Compute its inverse
        K_obv_obv_inv = np.linalg.inv(K_obv_obv)

        # Compute determinant of K
        K_obv_obv_det = np.linalg.det(K_obv_obv)

        # Estimate Prior mean
        mean = (one.T @ K_obv_obv_inv @ self._Y) / (one.T @ K_obv_obv_inv @ one)

        # Estimate Prior variance
        variance = ((self._Y - mean * one).T @ K_obv_obv_inv @ (self._Y - mean * one)) / n

        # Compute log-likelihood
        likelihood = -(n / 2) * np.log(variance) - 0.5 * np.log(K_obv_obv_det)

        # Update attributes (for .predict() later on)
        self._K_obv_obv, self._K_obv_obv_inv, self._mean, self._variance = K_obv_obv, K_obv_obv_inv, mean, variance

        return -likelihood.flatten()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fits the model, by using gradient-optimization approach (negative log-likelihood) to optimize the
        hyperparameters of the covariance function K(X1, X2 | thetas).

        Args:
            X (np.ndarray):
                Training dataset of shape (N, ?)

            Y (np.ndarray):
                Training dataset of shape (N, ?)
        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, -1)

        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, -1)

        # Calculates the thetas of the kernel function
        self._X, self._Y = X, Y

        # Generate random starting points (thetas)
        hp_types = self._get_kernel_hps()
        n = len(hp_types)  # number of hyperparameters for kernel function

        # Lower bound := 0, upper bound := 2
        lower_bound, upper_bound = 0.0, 2.0
        initial_thetas = lower_bound + np.random.rand(self._n_restarts, n) * (upper_bound - lower_bound)

        # Create the bounds for each algorithm
        bounds = Bounds([lower_bound] * n, [upper_bound] * n)

        # Run optimizer on all sampled thetas
        opt_para = np.zeros((self._n_restarts, n))
        opt_func = np.zeros((self._n_restarts, 1))
        for i in range(self._n_restarts):
            res = minimize(self.negative_likelihood, initial_thetas[i], method=self._optimizer, bounds=bounds)
            opt_para[i] = res.x  # extract the h
            opt_func[i] = res.fun  # negative likelihood
        # Locate the optimum results
        self._thetas = opt_para[np.argmin(opt_func)]

        # Update attributes with the best thetas
        self.negative_likelihood(self._thetas)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Inference method of the gaussian processes, to predict y-values of new (unseen) x-values.
        This method can only be called, after using the .fit() method.


        Args:
            X (np.ndarray): new unseen dataset

        Returns:
            tuple[np.ndarray, np.ndarray]: with the following information
                - [0] (np.ndarray): y-values
                - [1] (np.ndarray): uncertainty of the y-values
        """
        assert self._kernel is not None, "Use the method .fit() before calling this method!"

        if len(X.shape) == 1:
            X = np.expand_dims(X, -1)

        n = len(self._X)
        one = np.ones((n, 1))  # vector of ones

        # Construct covariance matrix between test and train data
        k = self.cov(self._X, X)

        # Mean prediction
        mean = self._mean + k.T @ self._K_obv_obv_inv @ (self._Y - self._mean * one)

        # Variance prediction
        cov = self._variance * (1 - k.T @ self._K_obv_obv_inv @ k)

        if self._return_cov:
            # Case: Return covariance matrix
            return mean, cov
        else:
            # Case: Return variance
            variance = np.diagonal(cov)
            return mean, variance

    def _get_kernel_class(self) -> Type[Kernel]:
        """
        Returns:
            Type[Kernel]: Class of the used kernel
        """
        kernel_mapping = {
            "linear": LinearKernel,
            "polynomial": PolynomialKernel,
            "sigmoid": SigmoidKernel,
            "laplacian": LaplacianKernel,
            "periodic": PeriodicKernel,
            "rbf": RBFKernel,
        }
        if self._kernel_method in kernel_mapping:
            return kernel_mapping[self._kernel_method]
        raise ValueError(f"Unknown kernel method {self._kernel_method}!")

    def _get_kernel_hps(self) -> dict:
        """
        Returns:
            dict: hyperparameter of the kernel function as dictionary
        """
        return self._get_kernel_class().get_hps()

    def _get_kernel(self, *thetas) -> Kernel:
        """
        Returns the kernel function of the gaussian process with the given hyperparameters (thetas)

        Args:
            *thetas: hyperparameters of the used kernel function

        Returns:
            Kernel: kernel function
        """
        kernel = self._get_kernel_class()
        return kernel(*thetas)
