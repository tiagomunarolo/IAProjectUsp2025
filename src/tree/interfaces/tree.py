from abc import ABC, abstractmethod

import numpy as np


class TreeInterface(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray): ...

    @abstractmethod
    def predict(self, x: np.ndarray): ...


class BaseTree(TreeInterface, ABC):

    def __init__(self, max_depth: int, min_samples_split: int = 2):
        """
        :param max_depth: Profundidade máxima da árvore
        :param min_samples_split:  Número mínimo de amostras necessárias para dividir um nó
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray): ...

    """
    Treinamento da arvore, dado um conjunto de dados X (preditor) e Y (saída)
    """

    @abstractmethod
    def predict(self, x: np.ndarray): ...

    """
    Predição da arvore, dado um conjunto de dados X (preditor)
    """
