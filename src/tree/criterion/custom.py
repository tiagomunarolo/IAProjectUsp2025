"""
Implementação aqui de um critério customizado
"""
import numpy as np
from sklearn.metrics import f1_score


def f1_proxy(y: np.ndarray):
    y = np.asarray(y)
    if len(np.unique(y)) == 1:
        return 0.0
    majority_class = np.bincount(y).argmax()
    predicted = np.full(len(y), majority_class)
    return f1_score(y, predicted)


def custom_criterion(**kwargs):
    size_left = kwargs['x_left'].shape[0]
    size_right = kwargs['x_right'].shape[0]
    total = size_left + size_right
    # critério gini a esquerda
    _, counts = np.unique(kwargs['y_left'], return_counts=True)
    probs = counts / counts.sum()
    gini_left = 1 - np.sum(probs ** 2)
    # critério gini a direita
    _, counts = np.unique(kwargs['y_right'], return_counts=True)
    probs = counts / counts.sum()
    gini_right = 1 - np.sum(probs ** 2)
    # critério gini geral
    gini = (gini_left * size_left + gini_right * size_right) / total
    # critério f1 a esquerda
    f1_left = f1_proxy(kwargs['y_left'].values)
    # critério f1 a direita
    f1_right = f1_proxy(kwargs['y_right'].values)
    # critério f1 geral
    f1 = (f1_left * size_left + f1_right * size_right) / total

    return gini * (1 - f1)
