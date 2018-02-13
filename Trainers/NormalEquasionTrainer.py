# coding=utf-8

from Models import ITrainer, ILinearRegression


class NormalEquasionTrainer(ITrainer):
    def __init__(self, model: ILinearRegression):
        super().__init__(model)

    def train(self, x, y):
        m, n = x.shape
        if m < 2:
            return [], []
        from scipy import linalg
        steps = [0, 1]
        errors = [self.model.score(x, y)]
        self.model.theta = linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        errors.append(self.model.score(x, y))
        return steps, errors
