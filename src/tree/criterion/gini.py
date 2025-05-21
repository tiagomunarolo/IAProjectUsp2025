import numpy as np

""""
Impureza Gini
"""


def gini_impurity(**kwargs):
    # gini impurity --> 1 - sum(p_i^2)
    # p_i = n_i / n
    # n_i = quantidade de elementos da classe i
    # n = total de elementos
    size_left = kwargs['x_left'].shape[0]
    _, counts = np.unique(kwargs['y_left'], return_counts=True)
    probs = counts / counts.sum()
    left = 1 - np.sum(probs ** 2)

    size_right = kwargs['x_right'].shape[0]
    _, counts = np.unique(kwargs['y_right'], return_counts=True)
    probs = counts / counts.sum()
    right = 1 - np.sum(probs ** 2)

    return (left * size_left + right * size_right) / (size_left + size_right)
