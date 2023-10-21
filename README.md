# PyGaussian
PyGaussian is a simple Python Framework for using Gaussian Processes (GPs). You can find more information about
GPs [here](https://en.wikipedia.org/wiki/Gaussian_process).

### Using GP with PyGaussian
For the following we want to train a GP to approximate the following function:
```python
def function_1D(X: np.ndarray) -> np.ndarray:
    y = (X * 6 - 2) ** 2 * np.sin(X * 12 - 4)

    return y
```

First we have to create our train and test data:
```python
# Training data
x_train = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ndmin=2).T
y_train = function_1D(x_train)

# Testing data
x_test = np.linspace(0.0, 1, 100).reshape(-1, 1)
```

Then we initialize our GP:
```python
from PyGaussian.model import GaussianProcess

model = GaussianProcess()
```

Now we have to use the `.fit()` to initialize the hyperparameters of our kernel method.
As default `GaussianProcess()` uses the [squared exponential kernel](https://wikimedia.org/api/rest_v1/media/math/render/svg/445ebf9ae2934e17d2dc4d9430f3e492391fd400).
```python
model.fit(x_train, y_train)
```

After fitting the model we can now use the GP to make inferences on new unseen data points. Notice that
GPs are stochastic models, so they give us their prediction as well as how uncertainty the prediction is.
```python
y_test, sigma = model.predict(x_test)
```

To summarize all up, the following plot shows how well the GP approximate the true function, given the few data points
which shows how sample efficient and highly interpretable GPs are.

![](gp_prediction.png)