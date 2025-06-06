"""
Implementação aqui de um critério customizado
"""
import numpy as np
from src.tree.metrics.metrics import f1_score


def f1_proxy(y: np.ndarray):
    y = np.asarray(y)
    if len(np.unique(y)) == 1:
        return 0.0
    majority_class = np.bincount(y).argmax()
    predicted = np.full(len(y), majority_class)
    return f1_score(y, predicted)


def f1_proxy_impurity(**kwargs):
    size_left = kwargs['y_left'].shape[0]
    size_right = kwargs['y_right'].shape[0]
    total = size_left + size_right
    # critério f1 a esquerda
    f1_left = f1_proxy(kwargs['y_left'])
    # critério f1 a direita
    f1_right = f1_proxy(kwargs['y_right'])
    # critério f1 geral
    f1 = (f1_left * size_left + f1_right * size_right) / total

    return 1 - f1
