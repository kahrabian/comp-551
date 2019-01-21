import numpy as np


class LinearRegression(object):
    def __init__(self):
        self._w = None

    @staticmethod
    def add_bs(x):
        bs_ft = np.ones((x.shape[0], 1))
        x_bs = np.concatenate((bs_ft, x), axis=1)

        return x_bs

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        assert self._w is not None

        x_bs = self.add_bs(x)
        y_prd = x_bs.dot(self._w)

        return y_prd
