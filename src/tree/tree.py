import numpy as np
import pandas as pd

from .interfaces import BaseTree


def gini_impurity(y):
    # gini impurity --> 1 - sum(p_i^2)
    # p_i = n_i / n
    # n_i = quantidade de elementos da classe i
    # n = total de elementos
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


class DecisionTree(BaseTree):
    """
    Implementação da árvore de decisão
    """

    def __init__(self, max_depth: int, min_samples_split: int = 2):
        super().__init__(max_depth, min_samples_split)
        self.num_features = None
        self.num_classes = None
        self.tree_ = None

    @staticmethod
    def _find_best_split(x: pd.DataFrame, y: np.ndarray) -> tuple:
        """
        Busca o melhor split, com base na impureza
        """
        # m = num de amostras, n = num de features
        m, n = x.shape
        # melhor feature e threshold
        best_feature, best_thresh = None, None
        # melhor critério de impureza
        best_criterion = float('inf')

        # para cada feature/atributo
        for index, feature in enumerate(x.columns.tolist()):
            thresholds = np.unique(x[feature])
            for index_t, t in enumerate(thresholds):
                print(f'feature: {index + 1}/{n}, threshold: {index_t + 1}/{len(thresholds)}')
                y_left = y[x[feature] <= t]  # seleciona todas linhas cujo valor feature <= t
                y_right = y[x[feature] > t]  # seleciona todas linhas cujo valor feature > t
                if len(y_left) == 0 or len(y_right) == 0:
                    # ignora splits que resultam em vazios
                    continue
                x_left = x[x[feature] <= t]
                x_right = x[x[feature] > t]
                # passa a árvore a esquerda e a direita para computar o critério
                score = gini_impurity(y)
                # Se o score de impureza for menor,isto é,
                # separamos bem os dados conforme o threshold, então
                # esse threshold eh o melhor
                if score < best_criterion:
                    best_criterion = score
                    best_feature = feature
                    best_thresh = t
        return best_feature, best_thresh

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int):
        """
        Construção da árvore de maneira recursiva
        :param x: features do dataset
        :param y: classes do dataset
        :param depth: profundidade da árvore
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        # criando o nó da árvore (classe predita = maior quantidade de amostras)
        node = {'predicted_class': predicted_class}
        if depth < self.max_depth:
            # enquanto não chegou na profundidade maxima
            # busca o melhor split
            feature, threshold = self._find_best_split(x, y)
            # melhor split selecionado
            if feature is not None:
                # separa os dados conforme o melhor split
                indices_left = x[feature] <= threshold
                x_left, y_left = x[indices_left], y[indices_left]
                x_right, y_right = x[~indices_left], y[~indices_left]
                # insere o melhor split na árvore de decisão
                node['feature'] = feature
                # insere o melhor threshold na arvore (para o atual split)
                node['threshold'] = threshold
                # cria a árvore a esquerda e a direita
                node['left'] = self._build_tree(x_left, y_left, depth + 1)
                node['right'] = self._build_tree(x_right, y_right, depth + 1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        # total de classes no dataset
        self.num_classes = len(set(y))
        # numero de features no dataset
        self.num_features = x.shape[1]
        # cria a árvore (profundidade inicial = 0)
        self.tree_ = self._build_tree(x, y, depth=0)
        return self

    @staticmethod
    def _predict_one(x: pd.DataFrame, node: dict):
        while 'feature' in node:
            feat = x[node['feature']]
            threshold = node['threshold']
            node = node['left'] if feat <= threshold else node['right']
        return node['predicted_class']

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        responses = []
        for index in range(x.shape[0]):
            responses.append(self._predict_one(x.iloc[index, :], self.tree_))
        return np.array(responses)
