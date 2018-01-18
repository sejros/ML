# coding=utf-8

from Measures.MSEMeasure import MSEMeasure, IMeasure
from Models import ILinearRegression
from Trainers.GradientDescent import GradientDescent
from Trainers.NormalEquasionTrainer import NormalEquasionTrainer, ITrainer

import scipy as sp


class MultivariateLinearRegression(ILinearRegression):
    """

    Syntax:
        >>> from utils import add_bias_column, mean_normalize
        >>> from Trainers.GradientDescent import GradientDescent
        >>> x = sp.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        >>> y = sp.array([1, 3, 5, 7])
        >>> x_, params = mean_normalize(x)
        >>> x_ = add_bias_column(x_)
        >>>
        >>> model = MultivariateLinearRegression(degree=2)
        >>> print(model.theta)
        [ 0.  0.  0.]
        >>> print(model.apply(x_))
        [ 0.  0.  0.  0.]
        >>> print(model.error(x_, y))
        10.5

        >>> model.trainer = GradientDescent(model=model, epochs=5, alpha=0.3)
        >>> model.train(x_, y)
        >>> print(model.theta)
        [ 3.32772     1.05831645  1.05831645]
        >>> print(model.apply(x_))
        [ 2.26940355  2.97494785  3.68049215  4.38603645]
        >>> print(model.error(x_, y))
        1.27323991891

    """
    def apply(self, x):
        m, n = x.shape
        assert n == self.degree, \
            "Количество столбцов не совпадает с порядком регрессии, m={}, n={}, degree={}".format(m, n, self.degree)
        return x.dot(self.theta)

    def gradient(self, x, y):
        m, n = x.shape

        y_ = self.apply(x)

        gradient = []

        for j in range(self.degree):
            gr = sum(sp.multiply((y_ - y), x[:, j])) / m
            gr += (self.regularization_rate * self.theta[j]) / m
            gradient.append(gr)
        gradient = sp.array(gradient)

        assert gradient.shape == (self.degree,)

        return gradient

    def init_weights(self):
        self.theta = sp.array([0.0] * self.degree)

    def __init__(self, degree,
                 trainer: ITrainer =None, measure: IMeasure =None, regularization: int =0.0):
        self.degree = degree + 1
        measure = measure or MSEMeasure(self)
        if degree < 10:
            trainer = trainer or NormalEquasionTrainer(model=self)
        else:
            trainer = trainer or GradientDescent(model=self)
        super().__init__(trainer=trainer, measure=measure, regularization=regularization)
