# Aqui pra gente implementar uma árvore de decisão personalizada

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))  # == 2, no nosso caso
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _custom_criterion(self, y_left, y_right):
        # novo critério!
        # Aqui q eu to pensando em implementar um modelo que se adapta ao problema
        # Exemplo: diferença absoluta da média da classe (para binária)
        def impurity(y):
            p = np.mean(y)
            return abs(0.5 - p)  # só como exemplo estranho

        return impurity(y_left) + impurity(y_right)

    def _best_split(self, X, y):
        m, n = X.shape
        best_feature, best_thresh = None, None
        best_criterion = float('inf')

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                y_left = y[X[:, feature] <= t]
                y_right = y[X[:, feature] > t]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                score = self._custom_criterion(y_left, y_right)
                if score < best_criterion:
                    best_criterion = score
                    best_feature = feature
                    best_thresh = t
        return best_feature, best_thresh

    def _build_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {
            'predicted_class': predicted_class
        }

        if depth < self.max_depth:
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] <= threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature'] = feature
                node['threshold'] = threshold
                node['left'] = self._build_tree(X_left, y_left, depth + 1)
                node['right'] = self._build_tree(X_right, y_right, depth + 1)
        return node

    def _predict_one(self, x, node):
        while 'feature' in node:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # dataset de cancer de mama
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # criando a arvore de decisão customizada
    tree = CustomDecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
