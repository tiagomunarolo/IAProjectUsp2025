import numpy as np
from sklearn.linear_model import LogisticRegression


def _get_score(x, y):
    if np.unique(y).shape[0] == 1:
        return 0
    lr = LogisticRegression()
    lr.fit(x, y)
    return 1 - lr.score(x, y)


def logistic_regression_impurity(**kwargs):
    x_left, x_right = kwargs['x_left'], kwargs['x_right']
    y_left, y_right = kwargs['y_left'], kwargs['y_right']

    left = _get_score(x_left, y_left)
    right = _get_score(x_right, y_right)

    return (left * x_left.shape[0] + right * x_right.shape[0]) / (x_left.shape[0] + x_right.shape[0])
