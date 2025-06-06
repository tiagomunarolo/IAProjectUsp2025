import numpy as np


def entropy(**kwargs):
    size_left = kwargs['y_left'].shape[0]
    _, counts_left = np.unique(kwargs['y_left'], return_counts=True)

    size_right = kwargs['y_right'].shape[0]
    _, counts_right = np.unique(kwargs['y_right'], return_counts=True)

    probs_left = counts_left / counts_left.sum()
    probs_right = counts_right / counts_right.sum()

    entropy_left = -np.sum(probs_left * np.log2(probs_left))
    entropy_right = -np.sum(probs_right * np.log2(probs_right))

    return (entropy_left * size_left + entropy_right * size_right) / (size_left + size_right)
