from typing import Type
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np

from PyGaussian.kernel import Kernel, GaussianKernel, PolynomialKernel


class GaussianProcess:
    """
    Class to represent the stochastic model Gaussian Processes.

    More information about how to implement a Gaussian Process can you find here:
    https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce#9f9c
    """

    def __init__(
            self,
            kernel_method: str = "gaussian",
            optimizer: str = "L-BFGS-B",
            n_restarts: int = 5,
    ):
        self._optimizer = optimizer
        self._n_restarts = n_restarts

        # Kernel function for calculating the covariance function K(X1, X2)
        self._kernel_method = kernel_method
        self._kernel = None
        self._thetas = None  # contains all hyperparameters for the kernel function

        # Dataset, used to train the hyperparameters theta
        self._X = None
        self._Y = None

        # Parameters for .predict() method
        self._K = None
        self._inv_K = None
        self._mean = None
        self._sigma = None

    def cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Returns the covariance matrix between X1 and X2, calculated by the covariance function K(X1, X2).

        Args:
            X1 (np.ndarray): dataset of shape (N1, M1)
            X2 (np.ndarray): dataset of shape (N2, M2)

        Returns:
            np.ndarray: covariance matrix of shape (N1, N2)
        """
        return np.array([[self._kernel(x1, x2) for x1 in X1] for x2 in X2])

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

        # Construct covariance matrix K and its inverse K^-1
        K = self.cov(self._X, self._X) + np.eye(n) * 1e-10
        inv_K = np.linalg.inv(K)

        # Estimate Prior mean
        mean = (one.T @ inv_K @ self._Y) / (one.T @ inv_K @ one)

        # Estimate Prior variance
        sigma = (self._Y - mean * one).T @ inv_K @ (self._Y - mean * one) / n

        # Compute determinant of K
        det_K = np.linalg.det(K)

        # Compute log-likelihood
        likelihood = -(n / 2) * np.log(sigma) - 0.5 * np.log(det_K)

        # Update attributes (for .predict() later on)
        self._K, self._inv_K, self._mean, self._sigma = K, inv_K, mean, sigma

        return -likelihood.flatten()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fits the model, by using gradient-optimization approach (negative log-likelihood) to optimize the
        hyperparameters of the covariance function K(X1, X2 | thetas).

        Args:
            X (np.ndarray): training dataset
            Y (np.ndarray): training dataset
        """
        # Calculates the thetas of the kernel function
        self._X, self._Y = X, Y

        # Generate random starting points (thetas)
        hp_types = self._get_kernel_hps()
        n = len(hp_types)  # number of hyperparameters for kernel function
        # Lower bound := -10, upper bound := 10
        initial_thetas = -100 + np.random.rand(self._n_restarts, n) * (100 - -100)

        # Create the bounds for each algorithm
        bounds = Bounds([-100] * n, [100] * n)

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

        n = len(self._X)
        one = np.ones((n, 1))  # vector of ones

        # Construct covariance matrix between test and train data
        k = self.cov(self._X, X)

        # Mean prediction
        mean = self._mean + k @ self._inv_K @ (self._Y - self._mean * one)

        # Variance prediction
        sigma = self._sigma * (1 - np.diag(k @ self._inv_K @ k.T))

        return mean.flatten(), sigma.flatten()

    def _get_kernel_class(self) -> Type[Kernel]:
        """
        Returns:
            Type[Kernel]: Class of the used kernel
        """
        if self._kernel_method == "polynomial":
            return PolynomialKernel
        elif self._kernel_method == "gaussian":
            return GaussianKernel
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
