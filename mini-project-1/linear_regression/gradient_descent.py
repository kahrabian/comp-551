import logging

import numpy as np

from data.helpers import calculate_mse, timeit
from .base import LinearRegression

logger = logging.getLogger(__name__)


class GradientDescentLinearRegression(LinearRegression):
    def __init__(self, beta, nu, eps):
        super().__init__()

        self._beta = beta
        self._nu = nu
        self._eps = eps

    def calculate_eps(self, x, y, w):
        err = np.linalg.norm(self.add_bs(x).dot(w) - y, ord=2)
        _err = np.linalg.norm(self.add_bs(x).dot(self._w) - y, ord=2)

        return abs(err - _err)

    @timeit
    def fit(self, x, y):
        assert self._w is None

        x_bs = self.add_bs(x)
        x_bs_tr = x_bs.transpose()

        x_bs_tr_y = x_bs_tr.dot(y)
        x_bs_tr_bs = x_bs_tr.dot(x_bs)

        beta = 1
        self._w = np.random.uniform(0, 1, size=(x_bs.shape[1],))

        epoch = 1
        while True:
            beta *= 1 + self._beta
            alpha = self._nu / beta

            w = self._w - 2 * alpha * (x_bs_tr_bs.dot(self._w) - x_bs_tr_y)
            eps = self.calculate_eps(x, y, w)
            self._w = w.copy()

            mse = calculate_mse(y, y_prd=self.predict(x))
            logger.info('[GD] E: {epoch}, MSE: {mse}'.format(epoch=epoch, mse=mse))

            if eps <= self._eps:
                break

            epoch += 1
