import numpy as np

from .base import LinearRegression


class ClosedFormLinearRegression(LinearRegression):
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        assert self._w is None

        x_bs = self.add_bs(x)
        x_bs_tr = x_bs.transpose()

        w = x_bs_tr.dot(x_bs)
        w = np.linalg.inv(w)
        w = w.dot(x_bs_tr)
        w = w.dot(y)

        self._w = w.copy()

        print(self._w)
