# coding=utf-8

from Models import IIterativeTrainer, IGradientedModel


class GradientDescent(IIterativeTrainer):
    """ Класс реализует метода градиентного спуска """
    def __init__(self, model: IGradientedModel, epochs: int =1000, alpha: float =0.1):
        super().__init__(model, epochs=epochs, alpha=alpha)

    def train(self, x, y):
        """ Метод проводит обучение модели на переданной обучающей выборке с ответами """
        steps = []
        errors = []
        i = 0
        while i < self.epochs:
            old_params = self.model.theta.copy()
            gradient = self.model.gradient(x, y)
            # print("Descent", old_params, gradient, self.model.error(x, y))
            new_params = old_params - self.learning_rate * gradient
            self.model.theta = new_params
            error = self.model.error(x, y)
            steps.append(i)
            errors.append(error)
            i += 1
        return steps, errors


# TODO реализовать стохастический градиентный спуск
"""
class StochasticGradientDescent(ITrainer):

    def train(self, x: sp.array, y: sp.array, epochs=10) -> tuple:
        m, n = x.shape
        steps = []
        errors = []
        i = 0
        while i < epochs:
            inner_error = []
            for j in range(m):
                old_params = self.model.theta.copy()
                gradient = self.model.gradient(x[j:j + 1], y[j:j + 1])
                new_params = old_params - self.learning_rate * gradient
                self.model.theta = new_params
                error = self.model.error(x[j:j+1], y[j:j+1])
                inner_error.append(error)
            steps.append(i)
            errors.append(sum(inner_error) / len(inner_error))
            i += 1
        return steps, errors
"""

# TODO реализовать пакетный градиентный спуск
