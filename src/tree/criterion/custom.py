"""
Implementação aqui de um critério customizado
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def mean_knn_distance(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return distances[:, 1:].mean()  # exclude self-distance


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

    dist_left = 0
    dist_right = 0
    if total < 5000:
        try:
            dist_left = mean_knn_distance(kwargs['x_left'], k=5)
            dist_right = mean_knn_distance(kwargs['x_right'], k=5)
        except ValueError:
            dist_left = 0
            dist_right = 0
    # critério gini geral
    gini = (gini_left * size_left + gini_right * size_right) / total
    modified_gini = gini * (1 / (1 + dist_left)) * (1 / (1 + dist_right))
    return modified_gini
