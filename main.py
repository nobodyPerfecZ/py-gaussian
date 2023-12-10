import matplotlib.pyplot as plt
import numpy as np

from PyGaussian.model import GaussianProcess


def function_1D(X):
    """1D Test Function"""
    y = (X * 6 - 2) ** 2 * np.cos(X * 12 - 4)
    return y


def function_1D_noisy(X):
    """1D Test Function with noise"""
    y = function_1D(X) + np.random.normal(0.1, 0.3, size=X.shape)
    return y


def linear_function_1D(X):
    """1D Linear Function"""
    y = (6 * X + 5)
    return y


def linear_function_1D_noisy(X):
    """1D Linear Function with noise"""
    y = linear_function_1D(X) + np.random.normal(0.1, 0.3, size=X.shape)
    return y


def plot_gaussian_process_with_uncertainty(X_true, Y_true, X_train, Y_train, X_test, Y_test, std):
    """ Plotting function for GPs by plotting uncertainty with the standard deviation. """
    fig, ax = plt.subplots()
    ax.plot(X_true, Y_true, "--", color="red", label="True Function")
    ax.plot(X_train, Y_train, "ro", label="Training Data")
    ax.plot(X_test, Y_test, "b-", label="GP Prediction")
    ax.fill_between(X_test, Y_test - 1.96 * std, Y_test + 1.96 * std, alpha=0.2, label="Uncertainty of GP prediction")
    ax.set_ylabel("f(x)")
    ax.set_xlabel("x")
    ax.legend()
    plt.show()


def plot_gaussian_process_functions(X_true, Y_true, X_train, Y_train, X_test, Y_test, cov, n_functions):
    """ Plotting function for GPs by plotting functions from the covariance. """
    fig, ax = plt.subplots()
    samples = np.random.multivariate_normal(Y_test, cov, size=n_functions)
    ax.plot(X_true, Y_true, "--", color="red", label="True Function")
    ax.scatter(X_train, Y_train, label='Observation', color='r', zorder=2)
    for sample in samples:
        ax.plot(X_test, sample, zorder=1)
    ax.set_ylabel("f(x)")
    ax.set_xlabel("x")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # True data
    x = np.linspace(0.0, 1, 100)
    y = function_1D(x)

    # Training data
    x_train = np.array([0, 0.1, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y_train = function_1D_noisy(x_train)

    # Testing data
    x_test = np.linspace(0.0, 1, 100)

    # Initialize the model
    model = GaussianProcess(kernel_method="periodic", n_restarts=20)
    model.fit(x_train, y_train)

    # Train the model
    y_test, cov = model.predict(x_test, return_cov=True)

    # Flatten the result
    std = np.sqrt(np.diagonal(cov))
    y_test = y_test.flatten()

    # Plot samples of functions from the covariance
    plot_gaussian_process_functions(x, y, x_train, y_train, x_test, y_test, cov, 20)

    # Plot the uncertainty of the gaussian process
    plot_gaussian_process_with_uncertainty(x, y, x_train, y_train, x_test, y_test, std)

