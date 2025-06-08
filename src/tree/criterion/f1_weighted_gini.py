"""
Implementação aqui de um critério customizado
"""
import numpy as np
from src.metrics.metrics import f1_score


def f1_proxy(y: np.ndarray):
    y = np.asarray(y)
    if len(np.unique(y)) == 1:
        # Se houver apenas uma classe, nenhuma impureza
        return 0
    majority_class = np.bincount(y).argmax()
    predicted = np.full(len(y), majority_class)
    return f1_score(y, predicted)


def f1_weighted_gini(**kwargs):
    # TODO: testar melhor esse critério
    size_left = kwargs['y_left'].shape[0]
    size_right = kwargs['y_right'].shape[0]
    total = size_left + size_right

    # critério gini a esquerda
    _, counts = np.unique(kwargs['y_left'], return_counts=True)
    probs = counts / counts.sum()
    weight_left = {0: 1.0, 1: 1.0}
    if len(probs) == 2:
        weight_left = {0: 1.0, 1: 5}
    weighted_probs = [weight_left[i] * p for i, p in enumerate(probs)]
    gini_left = 1 - np.sum(p ** 2 for p in weighted_probs)

    # critério gini a direita
    _, counts = np.unique(kwargs['y_right'], return_counts=True)
    probs = counts / counts.sum()
    weight_right = {0: 1.0, 1: 1.0}
    if len(probs) == 2:
        weight_right = {0: 1.0, 1: 5}
    weighted_probs = [weight_right[i] * p for i, p in enumerate(probs)]
    gini_right = 1 - np.sum(p ** 2 for p in weighted_probs)

    # critério gini geral
    gini = (gini_left * size_left + gini_right * size_right) / total
    # critério f1 a esquerda
    f1_left = f1_proxy(kwargs['y_left'])
    # critério f1 a direita
    f1_right = f1_proxy(kwargs['y_right'])
    # critério f1 geral
    f1 = (f1_left * size_left + f1_right * size_right) / total
    # dá uma ponderação de 80% para o gini e 20% para o f1
    return gini * 0.8 + 0.2 * (1 - f1)
