# PyGaussian
PyGaussian is a simple Python Framework for using Gaussian Processes (GPs). You can find more information about
GPs [here](https://en.wikipedia.org/wiki/Gaussian_process).

### Using GP with PyGaussian
For the following we want to train a GP to approximate the following function:
```python
def function_1D(X):
    """1D Test Function"""
    y = (X * 6 - 2) ** 2 * np.cos(X * 12 - 4)
    return y


def function_1D_noisy(X):
    """1D Test Function with noise"""
    y = function_1D(X) + np.random.normal(0.1, 0.3, size=X.shape)
    return y
```

First we have to create our train and test data:
```python
# True data (in reality not available to us)
x = np.linspace(0.0, 1, 100)
y = function_1D(x)

# Training data (observed with noise)
x_train = np.array([0, 0.1, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
y_train = function_1D_noisy(x_train)

# Testing data
x_test = np.linspace(0.0, 1, 100)
```

Then we initialize our GP:
```python
from PyGaussian.model import GaussianProcess

model = GaussianProcess(kernel_method="periodic", n_restarts=20)
```

Now we have to use the `.fit()` to initialize the hyperparameters of our kernel method.
As default `GaussianProcess()` uses the [squared exponential kernel](https://wikimedia.org/api/rest_v1/media/math/render/svg/445ebf9ae2934e17d2dc4d9430f3e492391fd400).
```python
model.fit(x_train, y_train)
```

After fitting the model we can now use the GP to make inferences on new unseen data points. Notice that
GPs are stochastic models, so they give us their prediction as well as how uncertainty the prediction is.
```python
y_test, cov = model.predict(x_test, return_cov=True)
```

To summarize all up, the following plots shows how well the GP approximate the true function, given the few data points
we have. It shows how sample efficient and highly interpreatable GPs are.

![](gaussian_uncertainty.png)

![](gaussian_functions.png)


### Future Features
The following list defines features, that are currently on work:

* [ ] Add more kernel functions (Matérn, Sum, ...) to PyGaussian