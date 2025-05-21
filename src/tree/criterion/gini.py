import numpy as np

""""
Impureza Gini
"""


def gini_impurity(**kwargs):
    # gini impurity --> 1 - sum(p_i^2)
    # p_i = n_i / n
    # n_i = quantidade de elementos da classe i
    # n = total de elementos
    _, counts = np.unique(kwargs['y'], return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)
