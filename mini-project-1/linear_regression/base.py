import numpy as np


class LinearRegression(object):
    def __init__(self):
        self._w = None

    @staticmethod
    def add_bs(x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        assert self._w is not None

        return self.add_bs(x).dot(self._w)
