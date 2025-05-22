import numpy as np
import pandas as pd
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
        self.num_classes = None
        self.tree_ = None
        self.criterion = Criterion.get_criterion(criterion)

    def _find_best_split(self, x: pd.DataFrame, y: np.ndarray) -> tuple:
        """
        Busca o melhor split, com base na impureza
        """
        # melhor feature e threshold
        best_feature, best_thresh = None, None
        # melhor critério de impureza
        best_criterion = float('inf')
        # para cada feature/atributo
        for index, feature in enumerate(x.columns.tolist()):
            thresholds = np.unique(x[feature])
            # limita o numero de thresholds para 1000
            if len(thresholds) > 100:
                thresholds = np.linspace(x[feature].min(), x[feature].max(), 100)
            for t in thresholds:
                y_left = y[x[feature] <= t]  # seleciona todas linhas cujo valor feature <= t
                y_right = y[x[feature] > t]  # seleciona todas linhas cujo valor feature > t
                if len(y_left) == 0 or len(y_right) == 0:
                    # ignora splits que resultam em vazios
                    continue
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    # ignora splits que resultam em folhas < min_samples_split
                    continue
                # passa a árvore a esquerda e a direita para computar o critério
                data = {'y_left': y_left,
                        'y_right': y_right,
                        'x_left': x[x[feature] <= t],
                        'x_right': x[x[feature] > t]}
                # Faz a soma ponderada das impurezas
                score = self.criterion(**data)
                # Se o score de impureza for menor,isto é,
                # separamos bem os dados conforme o threshold, então
                # esse threshold eh o melhor
                if score < best_criterion:
                    best_criterion = score
                    best_feature = feature
                    best_thresh = t
        return best_feature, best_thresh

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
        if depth < self.max_depth:
            # enquanto não chegou na profundidade maxima
            # busca o melhor split
            feature, threshold = self._find_best_split(x, y)
            # melhor split selecionado
            if feature is not None:
                logger.debug(f'Parent: {parent} | '
                             f'Depth = {depth} | Best split: [{feature}]: {threshold}')

                # separa os dados conforme o melhor split
                indices_left = x[feature] <= threshold
                x_left, y_left = x[indices_left], y[indices_left]
                x_right, y_right = x[~indices_left], y[~indices_left]
                # insere o melhor split na árvore de decisão
                node['feature'] = feature
                # insere o melhor threshold na arvore (para o atual split)
                node['threshold'] = threshold
                # cria a árvore a esquerda e a direita
                node['left'] = self._build_tree(x_left, y_left, depth + 1, 'left')
                node['right'] = self._build_tree(x_right, y_right, depth + 1, 'right')
        return node

    def fit(self, x: pd.DataFrame, y: pd.Series) -> 'DecisionTreeAdapted':
        """ Processo de trenamento da árvore de decisão"""
        # total de classes no dataset
        self.num_classes = len(set(y))
        # numero de features no dataset
        self.num_features = x.shape[1]
        # cria a árvore (profundidade inicial = 0)
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
