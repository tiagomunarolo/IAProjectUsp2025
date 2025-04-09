# Aqui pra gente implementar uma árvore de decisão personalizada

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 1


def gini_impurity(y):
    # gini impurity --> 1 - sum(p_i^2)
    # p_i = n_i / n
    # n_i = quantidade de elementos da classe i
    # n = total de elementos
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def custom_impurity_dummy(y):
    p = np.mean(y)
    return abs(0.5 - p)  # só como exemplo estranho


def impurity_to_research_svc(X, y):
    # objetivo é minimizar essa impureza
    # se só tiver uma classe, não há impureza
    if len(np.unique(y)) == 1:
        return 0
    # nesse caso, quanto melhor o modelo, menor a impureza
    # se o SVC conseguir prever tudo certo, a impureza eh 0
    # se o SVC prever tudo errado, a impureza eh 1
    # ou seja, 1 - acuracia
    # extremamente lento
    svc = SVC(kernel='linear')
    # acelerar o processo
    # seleciona as duas melhores features pro conjunto atual
    svc.fit(X, y)
    current_score = svc.score(X, y)
    return 1 - current_score


def impurity_to_research_svc(X, y):
    # objetivo é minimizar essa impureza
    # se só tiver uma classe, não há impureza
    if len(np.unique(y)) == 1:
        return 0
    # nesse caso, quanto melhor o modelo, menor a impureza
    # se o SVC conseguir prever tudo certo, a impureza eh 0
    # se o SVC prever tudo errado, a impureza eh 1
    # ou seja, 1 - acuracia
    # extremamente lento
    svc = SVC(kernel='linear')
    # acelerar o processo
    # seleciona as duas melhores features pro conjunto atual
    svc.fit(X, y)
    current_score = svc.score(X, y)
    return 1 - current_score

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree_ = None
        self.impurity_fn = impurity_to_research_svc

    def fit(self, X, y):
        self.n_classes_ = len(set(y))  # == 2, no nosso caso
        # seleciona os melhores 10 features
        X = SelectKBest(chi2, k=10).fit_transform(X, y)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _custom_criterion(self, y_left, x_left, y_right, x_right):
        # novo critério!
        # Aqui q eu to pensando em implementar um modelo que se adapta ao problema
        return self.impurity_fn(x_left, y_left) + self.impurity_fn(x_right, y_right)

    def _best_split(self, X, y):
        m, n = X.shape
        best_feature, best_thresh = None, None
        best_criterion = float('inf')

        for index, feature in enumerate(range(n)):  # para cada feature/atributo
            thresholds = np.unique(X[:, feature])
            # para cada threshold, pega valores unicos
            if len(thresholds) > 100:
                # cria um intervalo de thresholds
                thresholds = np.linspace(X[:, feature].min(), X[:, feature].max(), 100)
            for index_t, t in enumerate(thresholds):
                print(f'feature: {index + 1}/{n}, threshold: {index_t + 1}/{len(thresholds)}')
                y_left = y[X[:, feature] <= t]  # seleciona todas linhas cujo valor feature <= t
                y_right = y[X[:, feature] > t]  # seleciona todas linhas cujo valor feature > t
                if len(y_left) == 0 or len(y_right) == 0:
                    # ignora splits que resultam em vazios
                    continue
                x_left = X[X[:, feature] <= t]
                x_right = X[X[:, feature] > t]
                # passa a arvore a esquerda e a direita para computar o critério
                score = self._custom_criterion(y_left, x_left, y_right, x_right)
                # Se o score de impureza for menor,isto é,
                # separamos bem os dados conforme o threshold, então
                # esse threshold eh o melhor
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

        if depth < self.max_depth:  # enquanto não chegou na profundidade maxima
            feature, threshold = self._best_split(X, y)
            # melhor split selecionado
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
    # estratifica o dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    # criando a arvore de decisão customizada
    tree = CustomDecisionTreeClassifier(max_depth=MAX_DEPTH)
    default_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini')
    default_tree.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (custom):", accuracy_score(y_test, y_pred))
    print("Accuracy (default):", accuracy_score(y_test, default_tree.predict(X_test)))

"""
Nos testes:
- gini impurity --> accuracy: 0.92
- custom impurity --> accuracy: 0.60 (esperado, por ser um critério sem sentido)

A arvore padrão do sklearn ainda tem alguma otimização
com os mesmos parâmetros de profundidade e impureza, deu acurácia de 0.93

"""
