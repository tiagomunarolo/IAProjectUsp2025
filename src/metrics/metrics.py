import numpy as np


def accuracy(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    """
    Cálculo da acuracia
    """
    return np.mean(y_pred == y_real)


def confusion_matrix(y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
    """
    Cálculo da matriz de confusão
    """
    return np.array([[np.sum((y_pred == 0) & (y_real == 0)), np.sum((y_pred == 0) & (y_real == 1))],
                     [np.sum((y_pred == 1) & (y_real == 0)), np.sum((y_pred == 1) & (y_real == 1))]])


def precision(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    """
    Cálculo da precisão
    """
    # Cálculo de VP e FP
    vp = np.sum((y_pred == 1) & (y_real == 1))
    fp = np.sum((y_pred == 1) & (y_real == 0))

    # Precisão
    return vp / (vp + fp) if (vp + fp) > 0 else 0.0


def recall(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    """
    Cálculo do recall
    """
    # Cálculo de VP e FN
    vp = np.sum((y_pred == 1) & (y_real == 1))
    fn = np.sum((y_pred == 0) & (y_real == 1))

    # Recall
    return vp / (vp + fn) if (vp + fn) > 0 else 0.0


def f1_score(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    """
    Cálculo do F1
    """
    p = precision(y_pred, y_real)
    r = recall(y_pred, y_real)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
