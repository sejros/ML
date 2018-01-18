# coding=utf-8

import abc
import scipy as sp
from utils import train_test_split


class IModel:
    def __init__(self):
        self.theta = None

    @abc.abstractmethod
    def apply(self, x: sp.array) -> sp.array:
        """ Метод, возвращающий теоретический результат по переданным значениям факторов

        :return: Одномерный вектор теоретически предсказанных значений результирующей переменной
        :rtype: numpy.array

        :param x: Одномерный массив значений факторного признака
        :type x: numpy.array

        """
        pass


class ITrainer:
    def __init__(self, model: IModel):
        self.model = model

    @abc.abstractmethod
    def train(self, x, y) -> tuple:
        pass


class IMeasure:
    def __init__(self, model: IModel):
        self.model = model

    @abc.abstractmethod
    def error(self, x, y) -> float:
        pass


class IIterativeTrainer(ITrainer):
    def __init__(self, model: IModel, epochs: int, alpha: float =0.1):
        self.learning_rate = alpha
        self.epochs = epochs
        super().__init__(model)

    @abc.abstractmethod
    def train(self, x, y) -> tuple:
        pass


class ISupervisedModel(IModel):
    def __init__(self, trainer: ITrainer, measure: IMeasure):
        self.trainer = trainer
        self.measure = measure
        super().__init__()
        self.init_weights()

    def train(self, x, y):
        self.trainer.train(x, y)

    def error(self, x: sp.array, y: sp.array) -> float:
        """ Функция ошибки предсказания по заданному набору значений факторных
        переменных и заданному набору значений результативной переменной:

        J = (1/2m) * sum((y_ - y)**2)
        J = (1/2m) * sum((b0 + b1*x1 + b2*x2 - y)**2)

        :returns: Мера ошибки
        :rtype: float

        :param x: Одномерный массив значений факторного признака
        :type x: numpy.array

        :param y: Одномерный массив значений результативной переменной
        :type y: numpy.array
        """
        return self.measure.error(x, y)

    @abc.abstractmethod
    def init_weights(self):
        """ Метод инициализирует веса модели. Используется для обучения модели заново """
        pass

    def learning_curve(self, x, y, plot: bool =False) -> tuple:
        """ Метод строит простейший график динамики обучения """

        m, n = x.shape
        assert m == len(y), "Несоответствующие по длине наборы данных"

        self.init_weights()
        steps, errors = self.trainer.train(x=x, y=y)
        if plot:
            import matplotlib.pyplot as plt
            plt.title('Learning curve')
            plt.plot(steps, errors)
            plt.ylim(ymin=0.0)
            plt.show()

        return steps, errors

    def overfitting_curve(self, x, y,
                          averaging_factor: int = 50, plot: bool =False) -> tuple:
        """ Метод строит кривые обучения.

        Используется для диагностики моделей машиного обучения
        """

        m, n = x.shape
        assert m == len(y), "Несоответствующие по длине наборы данных"

        sizes_of_train_set = []
        train_errors = []
        test_errors = []
        batch_size = 1

        while batch_size <= m - 1:
            step_train_errors = []
            step_test_errors = []
            # print("Overfitting", m, batch_size, m - batch_size)
            for step in range(averaging_factor):
                train, test = train_test_split(m, m - batch_size)
                # print(step, train, test, x[train].shape, x[test].shape)

                self.init_weights()
                self.train(x[train], y[train])
                step_train_errors.append(self.error(x[train], y[train]))
                step_test_errors.append(self.error(x[test], y[test]))
            sizes_of_train_set.append(batch_size + 1)
            train_errors.append(sum(step_train_errors) / len(step_train_errors))
            test_errors.append(sum(step_test_errors) / len(step_test_errors))
            print("Overfitting curve step", batch_size, "of", m)
            batch_size += max(int(m / 50), 1)

        if plot:
            import matplotlib.pyplot as plt
            plt.title('Overfitting curve')
            plt.plot(sizes_of_train_set, train_errors, color='blue', label='Train error')
            plt.plot(sizes_of_train_set, test_errors, color='red', label='Validation error')
            plt.ylim(ymin=0.0)
            plt.xlabel('Size of a train set')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        return sizes_of_train_set, train_errors, test_errors


class IGradientedModel(ISupervisedModel):
    def __init__(self, trainer: ITrainer, measure: IMeasure, regularization: float):
        self.regularization_rate = regularization
        super().__init__(trainer=trainer, measure=measure)

    @abc.abstractmethod
    def apply(self, x):
        pass

    @abc.abstractmethod
    def gradient(self, x: sp.array, y: sp.array) -> float:
        """ Градиент ошибки

        Частные производные (градиент) функции ошибки:
        dJ/db0 = (1/m) * sum(y_ - y)
        dJ/dbi = (1/m) * sum((y_ - y) * xi)

        :return: Одномерный массив градиента ошибки модели
        :rtype: numpy.array

        :param x: Одномерный массив значений факторного признака
        :type x: numpy.array

        :param y: Одномерный массив значений результативной переменной
        :type y: numpy.array
        """
        pass

    @abc.abstractmethod
    def init_weights(self):
        pass

    def lambda_curve(self, x, y,
                     n: int = 25, plot: bool =False, test_size: int =0, averaging_factor: int =20) -> tuple:
        lambdas = []
        train_errors = []
        test_errors = []

        b = 10 ** (3 / (n + 1))

        for i in range(n):
            lambda_ = 1.0 / b**i
            step_train_errors = []
            step_test_errors = []
            for step in range(averaging_factor):
                train, test = train_test_split(len(x), test_size)
                self.init_weights()
                self.regularization_rate = lambda_
                self.train(x[train], y[train])
                step_train_errors.append(self.error(x[train], y[train]))
                step_test_errors.append(self.error(x[test], y[test]))
            lambdas.append(lambda_)
            train_errors.append(sum(step_train_errors) / len(step_train_errors))
            if test_size:
                test_errors.append(sum(step_test_errors) / len(step_test_errors))
            print("Lambda curve step", i, "of", n)

        if plot:
            import matplotlib.pylab as plt
            plt.title('Lambda curve')
            plt.semilogx(lambdas, train_errors, color='blue', label='Train error')
            if test_size:
                plt.semilogx(lambdas, test_errors, color='red', label='Validation error')
            plt.ylim(ymin=0.0)
            plt.xlabel('Regularization parameter')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        return lambdas, train_errors, test_errors

    def alpha_curve(self, x, y,
                    n: int = 25, plot: bool =False, test_size: int =0, averaging_factor: int =50) -> tuple:
        alphas = []
        train_errors = []
        test_errors = []

        b = 10 ** (3 / (n + 1))

        for i in range(n):
            alpha = 1.0 / b**i
            step_train_errors = []
            step_test_errors = []
            for step in range(averaging_factor):
                train, test = train_test_split(len(x), test_size)
                self.init_weights()
                self.trainer.learning_rate = alpha
                self.train(x[train], y[train])
                step_train_errors.append(self.error(x[train], y[train]))
                step_test_errors.append(self.error(x[test], y[test]))
            alphas.append(alpha)
            train_errors.append(sum(step_train_errors) / len(step_train_errors))
            if test_size:
                test_errors.append(sum(step_test_errors) / len(step_test_errors))
            print("Alpha curve step", i, "of", n)

        if plot:
            import matplotlib.pylab as plt
            plt.title('Alpha curve')
            plt.semilogx(alphas, train_errors, color='blue', label='Train error')
            if test_size:
                plt.semilogx(alphas, test_errors, color='red', label='Validation error')
            plt.ylim(ymin=0.0)
            plt.xlabel('Learning rate')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        return alphas, train_errors, test_errors


class ILinearRegression(IGradientedModel):
    """ Абстрактный базовый класс для регрессионных моделей

    Описывет общий интерфейс моделей - способность к прогнозу, оценке ошибки и градиенту.
    Относится к моделям и методам обучения с учитетелем, так как oценка ошибки и градиента
    требует указания целевых знечений показателя Y.
    """
    def __init__(self, trainer: ITrainer, measure: IMeasure, regularization: int):
        super().__init__(trainer=trainer, measure=measure, regularization=regularization)

    @abc.abstractmethod
    def apply(self, x):
        pass

    @abc.abstractmethod
    def gradient(self, x, y):
        pass

    @abc.abstractmethod
    def init_weights(self):
        pass


##############################################################


class SupportVectorMachines(IGradientedModel):
    def gradient(self, x, y):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError


class ANNAdapter(ISupervisedModel):
    def __init__(self, trainer: ITrainer, measure: IMeasure):
        super().__init__(trainer, measure)

    def init_weights(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError

    def train(self, x, y):
        raise NotImplementedError


class KNearestNeighbours(ISupervisedModel):
    def init_weights(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError

    def __init__(self, trainer: ITrainer, measure: IMeasure):
        super().__init__(trainer, measure)

