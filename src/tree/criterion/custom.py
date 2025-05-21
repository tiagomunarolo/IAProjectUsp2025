"""
Implementação aqui de um critério customizado
"""
import numpy as np
from scipy.spatial.distance import cdist


def custom_criterion(**kwargs):
    # O critério de associar um não

    size_left = kwargs['x_left'].shape[0]
    _, counts = np.unique(kwargs['y_left'], return_counts=True)
    probs = counts / counts.sum()
    left = 1 - np.sum(probs ** 2)

    size_right = kwargs['x_right'].shape[0]
    _, counts = np.unique(kwargs['y_right'], return_counts=True)
    probs = counts / counts.sum()
    right = 1 - np.sum(probs ** 2)

    _left = kwargs['x_left'].loc[kwargs['y_left'] == 1, :]
    _right = kwargs['x_right'].loc[kwargs['y_right'] == 1, :]

    mean_left = cdist(_left, _left, metric='euclidean').mean()
    mean_right = cdist(kwargs['x_right'], kwargs['x_right'], metric='euclidean').mean()

    return mean_left * mean_right * ((left * size_left + right * size_right) / (size_left + size_right))
