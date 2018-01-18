# coding=utf-8


from Classification.LogisticRegression import LogisticRegression
from Regressions.MultivariateLinearRegression import MultivariateLinearRegression, GradientDescent
from utils import mean_normalize, add_bias_column

import scipy as sp
from sklearn.datasets import load_iris

data = load_iris()
raw_features = sp.array(data.data).transpose()
raw_label = sp.array(data.target)
x = []

x = sp.array(raw_features).transpose()
x, params = mean_normalize(x)
x = add_bias_column(x)
y = raw_label.copy()

y[y != 2] = 0
y[y == 2] = 1

# print(x)
# print(y)

# model = MultivariateLinearRegression(degree=x.shape[-1]-1)
# model.trainer = GradientDescent(model=model, epochs=100)
# model.learning_curve(x, y, plot=True)

model = LogisticRegression(degree=x.shape[-1]-1)
model.learning_curve(x, y, plot=True)
model.overfitting_curve(x, y, plot=True)
model.alpha_curve(x, y, plot=True, test_size=30)
model.lambda_curve(x, y, plot=True, test_size=30)