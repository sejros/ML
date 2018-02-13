# coding=utf-8


import scipy as sp

from Models import ILinearRegression
from Measures.MSEMeasure import MSEMeasure, IMeasure
from Trainers.NormalEquasionTrainer import NormalEquasionTrainer, ITrainer


class UnivariateLinearRegression(ILinearRegression):
    """ Класс реализует модель парной линейной регрессии

    Syntax:
        >>> from utils import add_bias_column
        >>> x = sp.array([[0], [1], [2]])
        >>> y = sp.array([1, 3, 5])
        >>> x_ = add_bias_column(x)


        >>> model = UnivariateLinearRegression()
        >>> model.theta
        array([0, 0])

        >>> model.predict(x_)
        array([ 0.,  0.,  0.])

        >>> model.fit(x_, y)
        >>> print(model.theta)
        [ 1.  2.]
        >>> model.predict(x_)
        array([ 1.,  3.,  5.])

        >>> model.init_weights()
        >>> model.theta
        array([0, 0])
        >>> print(model.predict(x_))
        [ 0.  0.  0.]


        Использование клааса в сочетании с другим методом обучения - градиентный спуск

        >>> from Trainers.GradientDescent import GradientDescent

        >>> x = sp.array([[0], [1], [2]])
        >>> y = sp.array([1, 3, 5])
        >>> x_ = add_bias_column(x)

        >>> model = UnivariateLinearRegression()
        >>> model.trainer = GradientDescent(model=model, epochs=5, alpha=0.3)
        >>> print(model.score(x_, y))
        5.83333333333
        >>> model.fit(x_, y)
        >>> print(model.theta)
        [ 1.18524  1.86148]
        >>> print(model.score(x_, y))
        0.00748730933333

    """
    def predict(self, x):
        m, n = x.shape
        assert n == 2, "Два столбца (включая единичный) для парной ререссии (m={}, n={})".format(m, n)

        return x.dot(self.theta)

    def gradient(self, x, y):
        m, n = x.shape
        assert n == 2, "Два столбца для обучения нелинейной регрессии"

        y_ = self.predict(x)
        j0 = sum(y_ - y) / m
        j1 = sum(sp.multiply(y_ - y, x[:, -1])) / m

        gradient = sp.array([j0, j1])
        assert gradient.shape == (2,)

        return gradient

    def init_weights(self):
        self.theta = sp.array([0, 0])

    def __init__(self, trainer: ITrainer =None, measure: IMeasure =None, regularization: int =0.0):
        """ Параметры регрессии """
        measure = measure or MSEMeasure(self)
        trainer = trainer or NormalEquasionTrainer(model=self)
        super().__init__(measure=measure, trainer=trainer, regularization=regularization)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
