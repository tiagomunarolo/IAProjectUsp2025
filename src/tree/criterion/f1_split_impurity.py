from src.metrics.metrics import f1_score
import numpy as np


def f1_split_impurity(**kwargs):
    y_left, y_right = kwargs['y_left'], kwargs['y_right']
    
    pred_left = np.full(len(y_left), round(np.mean(y_left)))
    pred_right = np.full(len(y_right), round(np.mean(y_right)))

    f1_l = f1_score(y_left, pred_left)
    f1_r = f1_score(y_right, pred_right)

    size_l, size_r = len(y_left), len(y_right)
    total = size_l + size_r

    return 1 - ((f1_l * size_l + f1_r * size_r) / total)
