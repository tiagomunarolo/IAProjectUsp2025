import numpy as np
import pandas as pd
from loguru import logger
from collections import deque
from joblib import Parallel, delayed
from src.tree.splitter_cython import find_best_split_cython  # fazer build dos arquivos em cython
from src.tree.criterion import Criterion
from src.tree.interfaces import BaseTree


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

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> dict:
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
        # Enquanto houver itens na pilha
        while stack:
            item = stack.pop()
            x, y, depth, node = item['x'], item['y'], item['depth'], item['node']
            classes, counts = np.unique(y, return_counts=True)
            if not classes.size:
                continue
            # A classe mais prevalente é a predita
            node['predicted_class'] = int(classes[np.argmax(counts)])
            # Se chegou na profundidade maxima ou se não houver amostras suficientes para fazer um split
            if depth >= self.max_depth or len(x) < self.min_samples_split:
                continue
            # melhor split selecionado
            parallel = Parallel(n_jobs=-1)
            _delayed = delayed(find_best_split_cython)
            best_split_parallel = parallel(
                _delayed(x, y, i, self.min_samples_split, self.criterion) for i in range(self.num_features))
            # ordena os splits pela impureza
            best_split_parallel.sort(key=lambda key: key[0])
            score, threshold, feature_index = best_split_parallel[0]
            logger.debug(f'Melhor split: '
                         f'{self.feature_names[feature_index]} <= {threshold}, score: {score}')
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
