import warnings
from typing import Optional, Type, Union

import numpy as np
from scipy import optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular

from PyGaussian.kernel import (
    Kernel,
    LaplacianKernel,
    LinearKernel,
    PeriodicKernel,
    PolynomialKernel,
    RBFKernel,
    SigmoidKernel
)

GPR_CHOLESKY_LOWER = True


class GaussianProcess:
    """
    This class represents a gaussian process regression (GPR) model, which predicts new values and uncertainty
    (std/cov) of its predictions.

    The implementation is based on Algorithm 2.1 of [RW2006]_:
    https://gaussianprocess.org/gpml/chapters/RW.pdf

    Args:
        kernel_method (str, optional):
            The kernel to be used for computing the covariance matrix K(X1, X2 | thetas)

        initial_thetas (Union[list[float], np.ndarray], optional):
            The initial hyperparameters (thetas) for the given kernel

        noise (float, optional):
            The prior noise which gets added to the diagonal of the covariance matrix K(X1, X2)
            It can ensure numerical stability during the optimization and inference

        n_restarts (int, optional):
            The number of restarts of different hyperparameters (thetas) for the kernel optimization

        random_state (int, optional):
            The seed for the random number generator
    """

    def __init__(
            self,
            kernel_method: str = "rbf",
            initial_thetas: Optional[Union[list[float], np.ndarray]] = None,
            noise: float = 1e-10,
            n_restarts: int = 10,
            random_state: int = None,
    ):
        assert n_restarts >= 0, f"Illegal n_restarts {n_restarts}! The argument should be higher or equal to 0!"

        self._noise = noise
        self._n_restarts = n_restarts
        self._random_state = random_state
        self._rng = np.random.RandomState(random_state)

        # Kernel function for calculating the covariance function K(X1, X2)
        self._kernel_method = kernel_method
        self._initial_thetas = initial_thetas if initial_thetas is None else np.array(initial_thetas)
        self._kernel = None  # the kernel
        self._thetas = None  # contains all hyperparameters (thetas) of the given kernel function

        # Training Dataset, used for inference (.predict()) and training kernel hyperparameters (thetas)
        self._X = None
        self._Y = None

        # Parameters used for inference (.predict())
        self._L = None
        self._alpha = None

    def obj_func(self, thetas: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Returns the negative log-marginal-likelihood (lml) and the negative gradients of dlml/dthetas.

        Args:
            thetas (np.ndarray):
                Numpy array of shape (N_thetas,)
                The used hyperparameter of the kernel function K(X1, X2 | thetas)

        Returns:
            tuple[float, np.ndarray]:

                log_marginal_likelihood (float):
                    The negative log-marginal-likelihood as defined above

                gradients (np.ndarray):
                    Numpy array of shape (N_thetas,)
                    The negative gradients dlml/d_thetas as defined above
        """
        log_marginal_likelihood, gradients = self.log_mariginal_likelihood(thetas)
        return -log_marginal_likelihood, -gradients

    def log_mariginal_likelihood(self, thetas: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Returns the log-marginal-likelihood (lml) and the gradients of dlml/dthetas with the given thetas.

        The log-marginal-likelihood (lml) is defined as following:
            - log p(y | X, thetas) = - 1/2 y^T * (K + noise * I) * y - 1/2 * log |(K + noise * I)| - n/2 log 2*pi


        The corresponding gradient dlml/dthetas is:
            - dlml/d_thetas = 1/2 trace((K^-1 y * y^T * K^-1 - K^-1) * dK/d_thetas), where

        dK/d_thetas are the gradients of the used kernel function K(X1, X2 | thetas).

        Args:
            thetas (np.ndarray):
                Numpy array of shape (N_thetas,)
                The used hyperparameter of the kernel function K(X1, X2 | thetas)

        Returns:
            tuple[float, np.ndarray]:

                log_marginal_likelihood (float):
                    The log-marginal-likelihood as defined above

                gradients (np.ndarray):
                    Numpy array of shape (N_thetas,)
                    The gradients dlml/d_thetas as defined above
        """
        # Construct the kernel function K(X1, X2) with given thetas
        self._kernel = self._get_kernel(thetas)

        K, K_gradient = self._kernel(self._X, self._X, return_gradient=True)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K[np.diag_indices_from(K)] += self._noise

        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros_like(thetas)

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), self._Y, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is originally thought to be a (1, n_samples) row vector. However,
        # in multi-outputs, y is of shape (n_samples, 2) and we need to compute
        # y^T . alpha for each output, independently using einsum. Thus, it
        # is equivalent to:
        # for output_idx in range(n_outputs):
        #     log_likelihood_dims[output_idx] = (
        #         y_train[:, [output_idx]] @ alpha[:, [output_idx]]
        #     )
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self._Y, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        # Eq. 5.9, p. 114, and footnote 5 in p. 114
        # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
        # alpha is supposed to be a vector of (n_samples,) elements. With
        # multi-outputs, alpha is a matrix of size (n_samples, n_outputs).
        # Therefore, we want to construct a matrix of
        # (n_samples, n_samples, n_outputs) equivalent to
        # for output_idx in range(n_outputs):
        #     output_alpha = alpha[:, [output_idx]]
        #     inner_term[..., output_idx] = output_alpha @ output_alpha.T
        inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
        # compute K^-1 of shape (n_samples, n_samples)
        K_inv = cho_solve(
            (L, GPR_CHOLESKY_LOWER), np.identity(K.shape[0]), check_finite=False
        )
        # create a new axis to use broadcasting between inner_term and
        # K_inv
        inner_term -= np.expand_dims(K_inv, -1)  # K_inv[..., np.newaxis]
        # Since we are interested about the trace of
        # inner_term @ K_gradient, we don't explicitly compute the
        # matrix-by-matrix operation and instead use an einsum. Therefore
        # it is equivalent to:
        # for param_idx in range(n_kernel_params):
        #     for output_idx in range(n_output):
        #         log_likehood_gradient_dims[param_idx, output_idx] = (
        #             inner_term[..., output_idx] @
        #             K_gradient[..., param_idx]
        #         )
        log_likelihood_gradient_dims = 0.5 * np.einsum(
            "ijl,jik->kl", inner_term, K_gradient
        )
        # the log likehood gradient is the sum-up across the outputs
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
        return log_likelihood, log_likelihood_gradient

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fits the gaussian process, by optimizing the hyperparameters of the given kernel method via a gradient-based
        approach. It minimizes the negative log-marginal-likelihood (lml):
            - log p(y | X, thetas) = - 1/2 y^T * (K + noise * I) * y - 1/2 * log |(K + noise * I)| - n/2 log 2*pi,

        where the thetas are used to compute the covariance matrix K  after the kernel K(X, X | thetas).

        This method also stores all necessary objects, which are used for the inference (.predict()).

        Args:
            X (np.ndarray):
                Numpy Array of shape (N,) or (N, N_features)
                Training data points

            Y (np.ndarray):
                Numpy Array of shape (N,) or (N,)
                Training data points
        """
        if len(X.shape) == 1:
            # Case: Change the shape of X (N,) to (N, 1)
            X = np.expand_dims(X, -1)

        if len(Y.shape) == 1:
            # Case: Change the shape of Y (N,) to (N, 1)
            Y = np.expand_dims(Y, -1)

        # Safe the training data
        self._X, self._Y = X.copy(), Y.copy()

        # Generate random starting points for our thetas
        hp_types = self._get_kernel_hps()
        n_thetas = len(hp_types)  # number of hyperparameters for kernel function
        bounds = [(np.log(log_bound[0]), np.log(log_bound[1])) for log_bound in hp_types.values()]

        if self._n_restarts >= 1 and len(bounds) >= 1:
            # Case: Optimize thetas
            initial_thetas = np.zeros(shape=(self._n_restarts, n_thetas))
            for i, bound in enumerate(bounds):
                initial_thetas[:, i] = self._rng.uniform(low=bound[0], high=bound[1], size=(self._n_restarts,))

            # Run optimizer on all sampled thetas
            theta_star = None
            f_opt_star = np.inf
            for i in range(self._n_restarts):
                opt_res = optimize.minimize(
                    self.obj_func,
                    initial_thetas[i],
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds
                )
                theta_opt, f_opt = opt_res.x, opt_res.fun
                if f_opt < f_opt_star:
                    f_opt_star = f_opt
                    theta_star = theta_opt

            if theta_star is None:
                warnings.warn("No optimal theta was found! Change n_restarts=... or kernel_method=...!")
                theta_star = self._initial_thetas
        else:
            # Case: Take initial thetas and do not optimize the hyperparameters of the kernel
            theta_star = self._initial_thetas

        # Safe the best founded thetas
        self._thetas = theta_star

        # Update attributes with the best founded thetas
        self._kernel = self._get_kernel(self._thetas)

        # Compute the inverse matrix L, alpha
        K_obv_obv = self._kernel(self._X, self._X)
        K_obv_obv[np.diag_indices_from(K_obv_obv)] += self._noise
        self._L = cholesky(K_obv_obv, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        self._alpha = cho_solve(
            (self._L, GPR_CHOLESKY_LOWER),
            self._Y,
            check_finite=False,
        )

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
            return_cov: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the predicted y-values (mean) of new unseen data points (X).

        Additionally, you can also return the uncertainty in form of the standard deviation (std) or the covariance
        matrix (cov) of the predicted y-values (mean).

        This method should only be called, after using the .fit() method.

        Args:
            X (np.ndarray):
                Numpy Array with shape of (N,) or (N, N_features)
                New unseen data points (X_test)

            return_std (bool, optional):
                Controls if the standard deviation (std) of each prediction should be returned

            return_cov (bool, optional):
                Controls if the covariance matrix (cov) for the prediction should be returned

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:

                mean (np.ndarray):
                    Numpy Array with shape of (N,)
                    The predicted y-values (mean) of new unseen data points (X)

                cov (np.ndarray, optional):
                    Numpy Array with shape of (N, N)
                    The covariance matrix of the predicted y-values

                std (np.ndarray, optional):
                    Numpy Array with shape of (N,)
                    The standard deviation (std) of the predicted y-values
        """
        assert return_std and not return_cov or \
            not return_std and return_cov or \
               not return_std and not return_cov, \
            "You can only set return_std or return_cov, but not both!"
        assert self._kernel is not None, "Use the method .fit() before calling this method!"

        if len(X.shape) == 1:
            X = np.expand_dims(X, -1)

        # Compute K_obv,*, K_*,*
        K_obv_star = self._kernel(X, self._X)
        K_star_star = self._kernel(X, X)

        # Mean Prediction
        # Alg 2.1, page 19, line 4 -> f* = K(X_train, X_test) * alpha
        mean = K_obv_star @ self._alpha

        # Covariance Prediction
        # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
        V = solve_triangular(
            self._L, K_obv_star.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
        )
        cov = K_star_star - V.T @ V

        # Clip negative variances and set them to the noise
        cov = np.clip(cov, self._noise, np.inf)

        if return_cov:
            # Case: Return also the covariance (cov) matrix
            return mean, cov
        elif return_std:
            # Case: Return also the standard deviation (std)
            variance = np.diagonal(cov)
            std = np.sqrt(variance)
            return mean, std
        else:
            # Case: Only return the mean
            return mean

    def _get_kernel_class(self) -> Type[Kernel]:
        """
        Converts the given kernel method (str) into the specific kernel class (cls) to instantiate a new kernel.

        Returns:
            Type[Kernel]:
                Class instance of the kernel
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

    def _get_kernel_hps(self) -> dict[str, tuple[float, float]]:
        """
        Returns the kernel hyperparameters with their given log-bounds.

        Returns:
            dict[str, tuple[float, float]]:

                hyperparameters (str):
                    The name of the hyperparameter (theta)

                log_bounds (tuple[float, float]):
                    The log-bounds of possible values for the given hyperparameter (theta)
        """
        return self._get_kernel_class().get_hps()

    def _get_kernel(self, thetas: Optional[np.ndarray] = None) -> Kernel:
        """
        Returns the kernel function K(X1, X2 | thetas) with the given hyperparameters (thetas).

        Args:
            thetas (np.ndarray, optional):
                Hyperparameters of the kernel function

        Returns:
            Kernel:
                The kernel function K(X1, X2 | thetas)
        """
        kernel = self._get_kernel_class()

        if thetas is None:
            return kernel()
        else:
            return kernel(*thetas)
