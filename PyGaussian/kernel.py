from typing import Any
from abc import ABC, abstractmethod
import numpy as np
import math


class Kernel(ABC):
    """ Abstract class representing a Kernel function to produce the Covariance. """

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_hps() -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]: dictionary, where
                - key := name of hyperparameter
                - value := type of hyperparameter
        """
        pass


class LinearKernel(Kernel):
    """
    Class representing the linear kernel.

    The implementation follows the formula of the following page:
    https://en.wikipedia.org/wiki/Polynomial_kernel

    The linear Kernel is defined as:
        - K(X1, X2) = X1^T * X2
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance value K(X1, X2) between X1 and X2, after the Linear Kernel:
            - K(X1, X2) = X1^T * X2

        Args:
            X1 (np.ndarray):
                First data point of shape (N,)

            X2 (np.ndarray):
                Second data point of shape (N,)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        return X1 @ X2 + self.eps

    def __str__(self) -> str:
        return f"LinearKernel()"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {}


class PolynomialKernel(Kernel):
    """
    Class representing the polynomial kernel.

    The implementation follows the formula of the following page:
    https://en.wikipedia.org/wiki/Polynomial_kernel

    The polynomial Kernel is defined as:
        - K(X1, X2) = a * (X1^T * X2 + bias)^polynomial
    """
    # Reference: https://en.wikipedia.org/wiki/Polynomial_kernel
    def __init__(self, bias: float, polynomial: float, eps: float = 1e-6):
        self.bias = bias
        self.polynomial = polynomial
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance value K(X1, X2) between X1 and X2, after the Polynomial Kernel:
            - K(X1, X2) = a * (X1^T * X2 + bias)^polynomial

        Args:
            X1 (np.ndarray):
                First data point of shape (N,)

            X2 (np.ndarray):
                Second data point of shape (N,)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        # Polynomial Kernel K(X1, X2) = a * (X1^T * X2 + bias)^polynomial
        return ((X1 @ X2 + self.bias) ** self.polynomial) + self.eps

    def __str__(self) -> str:
        return f"PolynomialKernel(bias={self.bias}, polynomial={self.polynomial})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "bias": float,
            "polynomial": float,
        }


class SigmoidKernel(Kernel):
    """
    Class representing the sigmoid kernel.

    The implementation follows the formula of the following page:
    https://dataaspirant.com/svm-kernels/

    The sigmoid Kernel is defined as:
        - K(X1, X2) = tanh(a * X1^T * X2 + bias)
    """
    def __init__(self, alpha: float, bias: float, eps: float = 1e-6):
        self.alpha = alpha
        self.bias = bias
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance value K(X1, X2) between X1 and X2, after the Sigmoid Kernel:
            - K(X1, X2) = tanh(a * X1^T * X2 + bias)

        Args:
            X1 (np.ndarray):
                First data point of shape (N,)

            X2 (np.ndarray):
                Second data point of shape (N,)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        # Sigmoid Kernel K(X1, X2) = α * tanh(X1^T * X2 + bias)
        return np.tanh(self.alpha * (X1 @ X2) + self.bias) + self.eps

    def __str__(self) -> str:
        return f"SigmoidKernel(alpha={self.alpha}, bias={self.bias})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "alpha": float,
            "bias": float,
        }


class LaplacianKernel(Kernel):
    """
    Class representing the laplacian kernel.

    The implementation follows the formula of the following page:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html

    The Laplacian Kernel is defined as:
        - K(X1, X2) = exp(||X1 - X2|| / σ)
    """
    def __init__(self, length_scale: float, eps: float = 1e-6):
        self.length_scale = length_scale
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance value K(X1, X2) between X1 and X2, after the Laplacian Kernel:
            - K(X1, X2) = exp(-2 * sin^2(pi ||X1 - X2|| / p))

        Args:
            X1 (np.ndarray):
                First data point of shape (N,)

            X2 (np.ndarray):
                Second data point of shape (N,)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        return np.exp(-(np.linalg.norm(X1 - X2, ord=1)) / self.length_scale) + self.eps

    def __str__(self) -> str:
        return f"LaplacianKernel(length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "length_scale": float,
        }


class PeriodicKernel(Kernel):
    """
    Class representing the periodic kernel.

    The implementation follows the formula of the following page:
    https://peterroelants.github.io/posts/gaussian-process-kernels/#Periodic-kernel

    The periodic kernel is defined as:
        - K(X1, X2) = exp(-2 * sin^2(pi ||X1 - X2|| / p))
    """

    def __init__(self, length_scale: float, eps: float = 1e-6):
        self.length_scale = length_scale
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance value K(X1, X2) between X1 and X2, after the Periodic kernel:
            - K(X1, X2) = exp(-2 * sin^2(pi ||X1 - X2|| / p))

        Args:
            X1 (np.ndarray):
                First data point of shape (N,)

            X2 (np.ndarray):
                Second data point of shape (N,)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        return np.exp(-(2 * np.sin(np.pi * np.linalg.norm(X1 - X2, ord=1)) ** 2) / self.length_scale) + self.eps

    def __str__(self) -> str:
        return f"PeriodicKernel(length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "length_scale": float,
        }


class RBFKernel(Kernel):
    """
    Class representing the radial basis function (RBF) kernel.

    The implementation follows the formula of the following page:
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel

    The radial basis function (RBF) kernel is defined as:
        - K(X1, X2) = exp(-||X1 - X2||^2 / (2 * σ^2))
    """

    def __init__(self, length_scale: float, eps: float = 1e-6):
        self.length_scale = length_scale
        self.eps = eps

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Calculates the covariance K(X1, X2) between X1 and X2, after the RBF kernel:
            - K(X1, X2) = exp(-||X1 - X2||^2 / (2 * σ^2))

        Args:
            X1 (np.ndarray):
                First data point of shape (N)

            X2 (np.ndarray):
                Second data point of shape (N)

        Returns:
            float:
                Covariance value K(X1, X2)
        """
        return np.exp(-(np.linalg.norm(X1 - X2) ** 2) / (2 * self.length_scale ** 2)) + self.eps

    def __str__(self) -> str:
        return f"RBFKernel(length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "length_scale": float,
        }

# TODO: Implement more Kernel Functions: https://en.wikipedia.org/wiki/Gaussian_process
# Possible (stable) Kernel Functions:
# Matern Kernel: (...)
# Linear + RBF (ARD) Kernel: (...)
# Inverse Multi-Quadratic Kernel: (...)
# Cosine Similarity Kernel: (...)
