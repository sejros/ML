# coding=utf-8

from Models import IGradientedModel, ITrainer
from Trainers.GradientDescent import GradientDescent
from Measures.CrossEntropyError import IMeasure, CrossEntropyError

import scipy as sp


class LogisticRegression(IGradientedModel):
    """

    Syntax:
        >>> from utils import add_bias_column, mean_normalize
        >>> from Trainers.GradientDescent import GradientDescent
        >>> import scipy as sp

        >>> x = sp.array([[3.0], [2.0], [2.0], [1], [5], [4]])
        >>> y = sp.array([0, 0, 0, 0, 1, 1])
        >>> x_, params = mean_normalize(x)
        >>> x_ = add_bias_column(x_)

        >>> model = LogisticRegression(degree=1)
        >>> print(model.theta)
        [ 0.  0.]
        >>> print(model.predict(x_))
        [ 0.5  0.5  0.5  0.5  0.5  0.5]
        >>> print(model.score(x_, y))
        0.69314718056

        >>> model.trainer = GradientDescent(model=model, epochs=10, alpha=0.3)
        >>> model.fit(x_, y)
        >>> print(model.theta)
        [-0.36186044  0.40140421]
        >>> print(model.predict(x_))
        [ 0.4145626   0.39043447  0.39043447  0.36683071  0.46395435  0.4391077 ]
        >>> print(model.score(x_, y))
        0.595568467012

    """
    def predict(self, x):
        m, n = x.shape
        assert n == self.degree, "Количество столбцов не совпадает с порядком регрессии, m={}, n={}".format(m, n)
        z = x.dot(self.theta)
        return 1 / (1 + sp.exp(-z))

    def gradient(self, x, y):

        m, n = x.shape

        y_ = self.predict(x)

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

    def __init__(self, degree: int,
                 trainer: ITrainer =None, measure: IMeasure =None, regularization: float =0.0):
        self.degree = degree + 1
        trainer = trainer or GradientDescent(model=self)
        measure = measure or CrossEntropyError(model=self)
        super().__init__(trainer=trainer, measure=measure, regularization=regularization)
