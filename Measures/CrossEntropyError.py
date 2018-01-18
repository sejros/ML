# coding=utf-8

from Models import IMeasure

import scipy as sp


class CrossEntropyError(IMeasure):
    def error(self, x, y):
        m, n = x.shape
        assert m == len(y), "Несогласованные по длине наборы данных"
        if m == 0:
            return None

        y_ = self.model.apply(x)
        error = -y * sp.log(y_) - (1-y) * sp.log(1 - y_)
        # print("Error", error)
        return sum(error) / m
