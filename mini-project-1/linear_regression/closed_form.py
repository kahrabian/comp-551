import numpy as np

from data.helpers import timeit
from .base import LinearRegression


class ClosedFormLinearRegression(LinearRegression):
    def __init__(self):
        super().__init__()

    @timeit
    def fit(self, x, y):
        assert self._w is None

        x_bs = self.add_bs(x)
        x_bs_tr = x_bs.transpose()
        self._w = np.linalg.inv(x_bs_tr.dot(x_bs)).dot(x_bs_tr).dot(y)
