# coding=utf-8


import scipy as sp


def train_test_split(length: int, num: int =0) -> tuple:
    """ Функция используется для разбиения массивов на обучающую и тренировочную выборки.

    Использование:
        train, test = train_test_split(len(Y), frac=0.8)
        # X - двумерный массив, используем для разбиения вторую ось
        X_train, Y_train = X[:, train], Y[train]
        X_train, Y_train = X[:, test], Y[test]

    :param length: Количество элементов в исходном массиве
    :type length: int

    :param num:  Количество элементов в желаемом тестовом массиве
    :type num: int

    :return: Два битовых массива, соответствующих разбиению исходного на обучающий и тестовый
    :rtype: tuple

     """
    split_idx = length - num
    shuffled = sp.random.permutation(list(range(length)))
    train = sorted(shuffled[:split_idx])
    test = sorted(shuffled[split_idx:])
    return train, test


def add_bias_column(x: sp.array):
    m, n = x.shape
    bias = sp.ones((m, 1))
    return sp.append(bias, x, axis=1)


def mean_normalize(x: sp.array, params: dict =None):
    """
    >>> x = sp.array([[0.0], [1.0], [2.0], [3.0]])
    >>> print(x)
    >>> x_, params = mean_normalize(x)
    >>> print(x_)
    """
    m, n = x.shape
    res = sp.empty_like(x)
    res[:] = x

    # TODO реализовать нормализацию по заданным параметрам

    params = {'mean': [], 'max': [], 'min': []}

    for i in range(n):
        mean = sum(x[:, i]) / len(x[:, i])
        max_ = max(x[:, i])
        min_ = min(x[:, i])
        res[:, i] = (res[:, i] - mean) / (max_ - min_)
        params['mean'].append(mean)
        params['min'].append(min_)
        params['max'].append(max_)

    return res, params


def polynomial_addition_1d(x: sp.array, degree: int):
    m, n = x.shape
    assert n == 1, "Только однофакторное расширение"
    res = sp.empty_like(x)
    res[:] = x
    for j in range(degree - 1):
        res = sp.append(res, x ** (j+2), axis=1)
    return res
