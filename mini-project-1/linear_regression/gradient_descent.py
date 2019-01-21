import numpy as np

from .base import LinearRegression


class GradientDescentLinearRegression(LinearRegression):
    def __init__(self, beta, nu, eps):
        super().__init__()

        self._beta = beta
        self._nu = nu
        self._eps = eps

    def fit(self, x, y):
        assert self._w is None

        x_bs = self.add_bs(x)
        x_bs_tr = x_bs.transpose()

        beta = 1 + self._beta
        self._w = np.random.uniform(0, 0.1, size=(x_bs.shape[1],))
        while True:
            alpha = self._nu / beta
            beta *= 1 + self._beta

            w = x_bs_tr.dot(x_bs)
            w = w.dot(self._w)
            w -= x_bs_tr.dot(y)
            w = self._w - 2 * alpha * w

            mse = np.linalg.norm(self._w - w, ord=2)
            self._w = w.copy()

            if mse <= self._eps:
                break
