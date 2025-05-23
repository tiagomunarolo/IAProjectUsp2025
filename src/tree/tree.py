import numpy as np
import pandas as pd
from collections import deque
from joblib import Parallel, delayed
from typing import Callable
from src.tree.criterion import gini_impurity
from src.tree.criterion import custom_criterion
from src.tree.interfaces import BaseTree


class Criterion:
    @staticmethod
    def get_criterion(criterion: str) -> Callable:
        if criterion == 'custom':
            return custom_criterion

        # Default
        return gini_impurity


class DecisionTreeAdapted(BaseTree):
    """
    Implementação da árvore de decisão
    """

    def __init__(self, max_depth: int, min_samples_split: int = 2, criterion: str = 'gini'):
        super().__init__(max_depth, min_samples_split)
        self.num_features = None
        self.feature_names = None
        self.num_classes = None
        self.tree_ = None
        self.criterion = Criterion.get_criterion(criterion)

    def _find_best_split(self, feature_idx, x: np.ndarray, y: np.ndarray) -> tuple:
        best_thresh = None
        n_samples, n_features = x.shape
        feature = x[:, feature_idx]
        best_score = np.inf
        thresholds = np.unique(feature)
        if len(thresholds) > 100:
            thresholds = np.linspace(feature.min(), feature.max(), num=100, endpoint=False)
        # Pré-filtragem de thresholds ruins
        for threshold in thresholds:
            left_indices = feature <= threshold
            right_indices = ~left_indices

            n_left = np.count_nonzero(left_indices)
            n_right = n_samples - n_left

            if n_left < self.min_samples_split or n_right < self.min_samples_split:
                continue

            y_left = y[left_indices]
            y_right = y[right_indices]

            score = self.criterion(
                y_left=y_left,
                y_right=y_right,
                x_left=None,
                x_right=None
            )

            if score < best_score:
                best_score = score
                best_thresh = threshold

        return best_score, best_thresh, feature_idx

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int, parent: str = 'root'):
        """
        Construção da árvore de maneira iterativa
        :param x: features do dataset
        :param y: classes do dataset
        :param depth: profundidade da árvore
        """
        stack = deque()
        root = {
            'x': x,
            'y': y,
            'depth': depth,
            'node': {}
        }
        stack.append(root)

        while stack:
            item = stack.pop()
            x, y, depth, node = item['x'], item['y'], item['depth'], item['node']

            classes, counts = np.unique(y, return_counts=True)
            # A classe mais prevalente é a predita
            node['predicted_class'] = int(classes[np.argmax(counts)])
            # Se chegou na profundidade maxima ou se não houver amostras suficientes para fazer um split
            if depth >= self.max_depth or len(x) < self.min_samples_split:
                continue
            # melhor split selecionado
            best_split_parallel = Parallel(n_jobs=-1)(
                delayed(self._find_best_split)(i, x, y) for i in range(self.num_features))
            best_split_parallel.sort(key=lambda x: x[0])  # ordena os splits pela impureza
            _, threshold, feature_index = best_split_parallel[0]
            # separa os dados conforme o melhor split
            mask = x[:, feature_index] <= threshold
            x_left, y_left = x[mask], y[mask]
            x_right, y_right = x[~mask], y[~mask]
            # insere o melhor split na árvore de decisão
            node['feature'] = self.feature_names[feature_index]
            node['threshold'] = threshold
            node['left'] = {}
            node['right'] = {}
            # cria a árvore a esquerda e a direita
            stack.append({'x': x_left, 'y': y_left, 'depth': depth + 1, 'node': node['left']})
            stack.append({'x': x_right, 'y': y_right, 'depth': depth + 1, 'node': node['right']})

        return root['node']

    def fit(self, x: pd.DataFrame, y: pd.Series) -> 'DecisionTreeAdapted':
        """ Processo de trenamento da árvore de decisão"""
        # total de classes no dataset
        self.num_classes = len(set(y))
        # numero de features no dataset
        self.num_features = x.shape[1]
        # Nome das features
        self.feature_names = x.columns.to_list()
        # cria a árvore (profundidade inicial = 0)
        x = x.to_numpy()
        y = y.to_numpy()
        self.tree_ = self._build_tree(x, y, depth=0)
        return self

    @staticmethod
    def _predict_one(x: pd.DataFrame, node: dict) -> int:
        """ Predição para um dado X"""
        while 'feature' in node:
            feat = x[node['feature']]
            threshold = node['threshold']
            node = node['left'] if feat <= threshold else node['right']
        return node['predicted_class']

    def predict(self, x: pd.DataFrame) -> np.array:
        """ Predição para um conjunto de dados """
        responses = []
        for index in range(x.shape[0]):
            responses.append(int(self._predict_one(x.iloc[index, :], self.tree_)))
        return np.array(responses)
