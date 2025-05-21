"""
Implementação aqui de um critério customizado
"""
import numpy as np


def custom_criterion(**kwargs):
    size_left = kwargs['x_left'].shape[0]
    _, counts = np.unique(kwargs['y_left'], return_counts=True)
    probs = counts / counts.sum()
    left = 1 - np.sum(probs ** 2)

    size_right = kwargs['x_right'].shape[0]
    _, counts = np.unique(kwargs['y_right'], return_counts=True)
    probs = counts / counts.sum()
    right = 1 - np.sum(probs ** 2)

    return (left * size_left + right * size_right) / (size_left + size_right)
