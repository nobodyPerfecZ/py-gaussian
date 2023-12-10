from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np


class Kernel(ABC):
    """ Abstract class representing a kernel function to produce the covariance matrix K. """

    def __call__(
            self,
            X1: np.ndarray,
            X2: np.ndarray,
            return_gradient: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Computes the covariance matrix K(X1, X2 | thetas) of the given kernel function.

        Args:
            X1 (np.ndarray):
                Numpy array of shape (N_samples1, N_features)

            X2 (np.ndarray):
                Numpy array of shape (N_samples2, N_features)

            return_gradient (bool, optional):
                Controls if the gradients dK(X1,X2)/dthetas should be returned

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:

                K (np.ndarray):
                    Numpy array of shape (N_samples1, N_samples2)
                    The covariance matrix between X1 and X2

                K_gradients (np.ndarray, optional):
                    Numpy array of shape (N_samples1, N_samples2, N_thetas)
                    The gradients dK(X1, X2)/dthetas
        """
        # Compute the covariance matrix K
        K = np.array([[self._call(x1, x2) for x2 in X2] for x1 in X1])
        if return_gradient:
            # Case: Compute the gradients dK(X1, X2)/dthetas)
            K_gradient = np.array([[self._gradient(x1, x2) for x2 in X2] for x1 in X1])
            if K_gradient.ndim == 2:
                # Case: Convert K_gradient of shape (N_samples1, N_samples2) into (N_samples1, N_samples2, 1)
                K_gradient = np.expand_dims(K_gradient, -1)
            return K, K_gradient
        else:
            return K

    @abstractmethod
    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        Computes the (single) value K(x1, x2) of the covariance matrix K.

        Args:
            X1 (np.ndarray):
                Numpy array of shape (N_features,)

            X2 (np.ndarray):
                Numpy array of shape (N_features,)

        Returns:
            float:
                The covariance value K(x1, x2) of the covariance matrix K.
        """
        pass

    @abstractmethod
    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        """
        Computes the (single) gradient dK(x1, x2)/dthetas of the covariance matrix K.

        Args:
            X1 (np.ndarray):
                Numpy array of shape (N_features,)

            X2 (np.ndarray):
                Numpy array of shape (N_features,)

        Returns:
            Union[float, np.ndarray]:
                The gradients dK(x1, x2)/dthatas of the covariance matrix K.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        """
        Returns the name of the hyperparameters (thetas) and the corresponding log-bounds as dictionary.

        Returns:
            dict[str, Any]:

                hyperparameters (str):
                    The name of the hyperparameter (theta)

                log_bounds (tuple[float, float]):
                    The log-bounds of possible values for the given hyperparameter (theta)
        """
        pass


class LinearKernel(Kernel):
    """
    This class represents the basic linear kernel.

    The implementation follows the formula of the following page:
    https://en.wikipedia.org/wiki/Polynomial_kernel

    The linear Kernel is defined as:
        - K(X1, X2) = X1^T * X2
    """

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        return X1 @ X2

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        return 0

    def __str__(self) -> str:
        return f"LinearKernel()"

    @staticmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        return {}


class PolynomialKernel(Kernel):
    """
    This class represents the polynomial kernel.

    The implementation follows the formula of the following page:
    https://en.wikipedia.org/wiki/Polynomial_kernel

    The polynomial kernel is defined as:
        - K(X1, X2) = (X1^T * X2 + b)^p

    The corresponding gradients dK(X1, X2)/dthetas are defined as:
        - dK(X1, X2)db = p * (X1^T * X2 + b)^p-1
        - dK(X1, X2)dp = (X1^T * X2 + b)^p * log(X1^T * X2 + b)
    """
    def __init__(self, bias: float = 1.0, polynomial: float = 1.0):
        self.bias = bias
        self.polynomial = polynomial

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        return (X1 @ X2 + self.bias) ** self.polynomial

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        gradient_bias = self.polynomial * ((X1 @ X2 + self.bias) ** (self.polynomial-1))
        gradient_polynomial = self._call(X1, X2) * np.log((X1 @ X2) + self.bias)
        return np.array([gradient_bias, gradient_polynomial])

    def __str__(self) -> str:
        return f"PolynomialKernel(bias={self.bias}, polynomial={self.polynomial})"

    @staticmethod
    def get_hps() -> dict[str, Any]:
        return {
            "bias": (1e-5, 1e5),
            "polynomial": (1e-5, 1e5),
        }


class SigmoidKernel(Kernel):
    """
    This class represents the sigmoid kernel.

    The implementation follows the formula of the following page:
    https://dataaspirant.com/svm-kernels/

    The sigmoid kernel is defined as:
        - K(X1, X2) = tanh(a * X1^T * X2 + b)

    The corresponding gradients dK(X1, X2)/dthetas are defined as:
        - dK(X1, X2)da = X1^T * X2 / cosh^2(a * X1^T * X2 + b)
        - dK(X1, X2)db = 1 / cosh^2(a * X1^T * X2 + b)
    """
    def __init__(self, alpha: float = 1.0, bias: float = 0.0):
        self.alpha = alpha
        self.bias = bias

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        return np.tanh(self.alpha * (X1 @ X2) + self.bias)

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        gradient_alpha = (X1 @ X2) / np.cosh(self.alpha * (X1 @ X2) + self.bias) ** 2
        gradient_bias = 1 / np.cosh(self.alpha * (X1 @ X2) + self.bias) ** 2
        return np.array([gradient_alpha, gradient_bias])

    def __str__(self) -> str:
        return f"SigmoidKernel(alpha={self.alpha}, bias={self.bias})"

    @staticmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        return {
            "alpha": (1e-5, 1e5),
            "bias": (1e-5, 1e5),
        }


class LaplacianKernel(Kernel):
    """
    This class represents the laplacian kernel.

    The implementation follows the formula of the following page:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html

    The laplacian kernel is defined as:
        - K(X1, X2) = exp(||X1 - X2|| / l)

    The corresponding gradients dK(X1, X2)/dthetas are defined as:
        - dK(X1,X2)/dl = K(X1, X2) * ||X1 - X2|| / l^2
    """
    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        return np.exp(-(np.linalg.norm(X1 - X2, ord=1)) / self.length_scale)

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        return self._call(X1, X2) * np.linalg.norm(X1 - X2, ord=1) / (self.length_scale ** 2)

    def __str__(self) -> str:
        return f"LaplacianKernel(length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        return {
            "length_scale": (1e-5, 1e5),
        }


class PeriodicKernel(Kernel):
    """
    This class represents the periodic kernel.

    The implementation follows the formula of the following page:
    https://www.cs.toronto.edu/~duvenaud/cookbook/

    The periodic kernel is defined as:
        - K(X1, X2) = exp(-(2 * sin^2(pi *(||X1-X2|| / p)) / l^2)

    The corresponding gradients dK(X1, X2)/dthetas are defined as:
        - dK(X1, X2)/dp = K(X1, X2) * -4/l^2 * pi * ||X1-X2|| * cos(pi * ||X1-X2|| / p) * sin(pi * ||X1-X2|| / p)
        - dK(X1,X2)/dl = K(X1, X2) * 4 * sin^2(pi * ||X1-X2|| / p)) / l^3
    """

    def __init__(self, period: float = 1.0, length_scale: float = 1.0):
        self.period = period
        self.length_scale = length_scale

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        return np.exp(-(2 * np.sin(np.pi * np.linalg.norm(X1 - X2, ord=1) / self.period) ** 2) / self.length_scale ** 2)

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        gradient_period = self._call(X1, X2) * \
                          (-4 / self.length_scale ** 2) * \
                          (np.pi * np.linalg.norm(X1 - X2, ord=1)) * \
                          np.cos(np.pi * np.linalg.norm(X1 - X2, ord=1) / self.period) * \
                          np.sin(np.pi * np.linalg.norm(X1 - X2, ord=1) / self.period)
        gradient_length_scale = self._call(X1, X2) * \
                                ((4 * np.sin(np.pi * np.linalg.norm(X1 - X2, ord=1) / self.period) ** 2) / (self.length_scale ** 3))
        return np.array([gradient_period, gradient_length_scale])

    def __str__(self) -> str:
        return f"PeriodicKernel(period={self.period}, length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        return {
            "period": (1e-5, 1e5),
            "length_scale": (1e-5, 1e5),
        }


class RBFKernel(Kernel):
    """
    This class represents the radial basis function (RBF) kernel.

    The implementation follows the formula of the following page:
    https://scikit-learn.org/stable/modules/gaussian_process.html#radial-basis-function-rbf-kernel

    The radial basis function (RBF) kernel is defined as:
        - K(X1, X2) = exp(-||X1 - X2||^2 / (2 * l^2))

    The corresponding gradients dK(X1, X2)/dthetas are defined as:
        - dK(X1,X2)/dl = K(X1, X2) + ||X1 - X2||^2 / l^3
    """

    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def _call(self, X1: np.ndarray, X2: np.ndarray) -> float:
        k = np.exp(-(np.linalg.norm(X1 - X2) ** 2) / (2 * self.length_scale ** 2))
        return k

    def _gradient(self, X1: np.ndarray, X2: np.ndarray) -> Union[float, np.ndarray]:
        k_gradient = self._call(X1, X2) * np.linalg.norm(X1 - X2) ** 2 / self.length_scale ** 3
        return k_gradient

    def __str__(self) -> str:
        return f"RBFKernel(length_scale={self.length_scale})"

    @staticmethod
    def get_hps() -> dict[str, tuple[float, float]]:
        return {
            "length_scale": (1e-5, 1e5),
        }

# TODO: Implement more Kernel Functions: https://en.wikipedia.org/wiki/Gaussian_process
# Possible (stable) Kernel Functions:
# Matern Kernel: (...)
# Sum Kernel: (...)
# Linear + RBF (ARD) Kernel: (...)
# Inverse Multi-Quadratic Kernel: (...)
# Cosine Similarity Kernel: (...)
# ...
