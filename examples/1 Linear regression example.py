# coding=utf-8


import pandas

from Regressions.MultivariateLinearRegression import MultivariateLinearRegression, GradientDescent
from utils import add_bias_column, polynomial_addition_1d

""" Импорт исходных данных из файлов """

dj = pandas.read_csv("../data/D&J-IND_101001_171001.txt")
gasp = pandas.read_csv("../data/GAZP_101001_171001.txt")
yndx = pandas.read_csv("../data/YNDX_101001_171001.txt")

""" Предварительная обработка данных """

res = pandas.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GASP'])
res1 = pandas.merge(res, yndx, on='<DATE>', suffixes=['_1', '_YNDX'])
y = res1['<CLOSE>_DJ']
x1 = res1['<CLOSE>_GASP']
x2 = res1['<CLOSE>']

"""Нормализация данных """

x1 = (x1 - min(x1)) / (max(x1) - min(x1))
x2 = (x2 - min(x2)) / (max(x2) - min(x2))
y = (y - min(y)) / (max(y) - min(y))

""" Подготовка факторов """

x = pandas.concat([x1, x2], axis=1).as_matrix()
x = add_bias_column(x)

""" Построение и обучение модели """

model = MultivariateLinearRegression(degree=2)
model.learning_curve(x, y, plot=True)
model.overfitting_curve(x, y, plot=True)
model.alpha_curve(x, y, plot=True)
model.lambda_curve(x, y, plot=True)

model = MultivariateLinearRegression(degree=2)
model.trainer = GradientDescent(model=model, epochs=100)
model.learning_curve(x, y, plot=True)
model.overfitting_curve(x, y, plot=True)
model.alpha_curve(x, y, plot=True, test_size=100)
model.lambda_curve(x, y, plot=True, test_size=100)

model = MultivariateLinearRegression(degree=5)
model.trainer = GradientDescent(model=model, epochs=100)
x1_poly = polynomial_addition_1d(x1.reshape((len(x1), 1)), 5)
x1_poly = add_bias_column(x1_poly)
model.learning_curve(x1_poly, y, plot=True)
model.overfitting_curve(x1_poly, y, plot=True)
model.alpha_curve(x1_poly, y, plot=True, test_size=100)
