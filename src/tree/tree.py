import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from loguru import logger
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
        # Usa amostragem inteligente ao invés de np.unique
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
        Construção da árvore de maneira recursiva
        :param x: features do dataset
        :param y: classes do dataset
        :param depth: profundidade da árvore
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        # criando o nó da árvore (classe predita = maior quantidade de amostras)
        node = {'predicted_class': int(predicted_class)}
        if depth < self.max_depth and y.shape[0] >= self.min_samples_split:
            # enquanto não chegou na profundidade maxima
            results = Parallel(n_jobs=-1)(delayed(self._find_best_split)(i, x, y) for i in range(x.shape[1]))
            results.sort(key=lambda x: x[0])  # sort by score
            _, threshold, feature_index = results[0]
            if threshold is None:
                return node
            # melhor split selecionado
            logger.debug(f'Parent: {parent} | '
                         f'Depth = {depth} | '
                         f'Best split: [{self.feature_names[feature_index]}]: {threshold}')

            # separa os dados conforme o melhor split
            mask = x[:, feature_index] <= threshold
            x_left, y_left = x[mask], y[mask]
            x_right, y_right = x[~mask], y[~mask]
            # insere o melhor split na árvore de decisão
            node['feature'] = self.feature_names[feature_index]
            # insere o melhor threshold na arvore (para o atual split)
            node['threshold'] = threshold
            # cria a árvore a esquerda e a direita
            _tree_data = [[x_left, y_left, depth + 1, 'left'], [x_right, y_right, depth + 1, 'right']]
            response = Parallel(n_jobs=-1)(
                delayed(self._build_tree)(x, y, depth, side) for x, y, depth, side in _tree_data)
            node['left'] = response[0]
            node['right'] = response[1]
            node.pop('predicted_class')
        return node

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
    def _predict_one(x: pd.DataFrame, node: dict):
        """ Predição para um dado X"""
        while 'feature' in node:
            feat = x[node['feature']]
            threshold = node['threshold']
            node = node['left'] if feat <= threshold else node['right']
        return node['predicted_class']

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """ Predição para um conjunto de dados """
        responses = []
        for index in range(x.shape[0]):
            responses.append(int(self._predict_one(x.iloc[index, :], self.tree_)))
        return np.array(responses)
