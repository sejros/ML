# coding=utf-8

import scipy as sp
from Models import IMeasure


class MSEMeasure(IMeasure):
    def error(self, x, y):
        m, n = x.shape
        if m == 0:
            return None

        y_ = self.model.apply(x)
        # print(y_.shape, y.shape)
        error = sp.square(y_ - y)
        # print(y_.shape, x.shape, y.shape, error.shape)
        assert error.shape == (m,)

        error = sum(error) / (2 * m)

        # print("MSE", error.shape, error)

        return error
