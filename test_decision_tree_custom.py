# ==========================================
# Custom Decision Tree com Critério Alternativo (inclui função dummy)
# ==========================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

MAX_DEPTH = 2  # profundidade máxima da árvore

# ----------------------
# Critérios de impureza
# ----------------------

def gini_impurity(y):
    # Cálculo da impureza de Gini (tradicional)
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def cost_sensitive_impurity(y_true, y_pred, cost_fn=5, cost_fp=1):
    # Impureza baseada em custo: penaliza mais falsos negativos
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return cost_fn * FN + cost_fp * FP

def surrogate_logreg_impurity(X, y):
    # Usa uma regressão logística como modelo auxiliar para medir impureza
    # Quanto menor a acurácia do modelo, maior a impureza do conjunto
    if len(np.unique(y)) == 1:
        return 0  # Se só tem uma classe, não há impureza
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return 1 - model.score(X, y)

def custom_impurity_dummy(X, y):
    # Função artificial de impureza: penaliza nós que estão muito longe do equilíbrio entre as classes
    if len(np.unique(y)) == 1:
        return 0
    p = np.mean(y)
    return abs(0.5 - p)

# ----------------------
# Classe da Árvore Customizada
# ----------------------
class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, impurity_fn=custom_impurity_dummy):
        self.max_depth = max_depth
        self.tree_ = None
        self.impurity_fn = impurity_fn  # função de impureza customizável

    def fit(self, X, y):
        # Define o número de classes no problema
        self.n_classes_ = len(set(y))
        # Seleciona as 10 melhores features com base no teste qui-quadrado
        X = SelectKBest(chi2, k=min(10, X.shape[1])).fit_transform(X, y)
        # Inicia o treinamento da árvore
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _custom_criterion(self, y_left, x_left, y_right, x_right):
        # Combina a impureza das divisões esquerda e direita
        return self.impurity_fn(x_left, y_left) + self.impurity_fn(x_right, y_right)

    def _best_split(self, X, y):
        # Busca exaustiva pelo melhor split (feature + threshold)
        m, n = X.shape
        best_feature, best_thresh = None, None
        best_criterion = float('inf')

        for index, feature in enumerate(range(n)):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 100:
                thresholds = np.linspace(X[:, feature].min(), X[:, feature].max(), 100)

            for index_t, t in enumerate(thresholds):
                # Divide os dados com base no threshold
                y_left = y[X[:, feature] <= t]
                y_right = y[X[:, feature] > t]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                x_left = X[X[:, feature] <= t]
                x_right = X[X[:, feature] > t]
                # Calcula a impureza combinada das duas partições
                score = self._custom_criterion(y_left, x_left, y_right, x_right)
                # Atualiza se encontrou melhor divisão
                if score < best_criterion:
                    best_criterion = score
                    best_feature = feature
                    best_thresh = t
        return best_feature, best_thresh

    def _build_tree(self, X, y, depth):
        # Cria o nó corrente e define a classe majoritária para ele
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {
            'predicted_class': predicted_class
        }

        # Se ainda não atingiu a profundidade máxima, tenta dividir o nó
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
        # Caminha pela árvore até chegar numa folha
        while 'feature' in node:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

    def predict(self, X):
        # Aplica a predição para todas as amostras
        return np.array([self._predict_one(x, self.tree_) for x in X])

# ----------------------
# Execução de teste com dataset real
# ----------------------
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Carregamento do dataset de câncer de mama
    X, y = load_breast_cancer(return_X_y=True)
    # Divisão treino/teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Instancia os modelos
    tree = CustomDecisionTreeClassifier(max_depth=MAX_DEPTH, impurity_fn=custom_impurity_dummy)
    default_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini')

    # Treinamento
    default_tree.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    # Predição
    y_pred_custom = tree.predict(X_test)
    y_pred_default = default_tree.predict(X_test)

    # Avaliação
    print("\n[RESULTADOS COM FUNÇÃO DUMMY]")
    print("Acurácia (custom):", accuracy_score(y_test, y_pred_custom))
    print("F1-score (custom):", f1_score(y_test, y_pred_custom))
    print("Recall (custom):", recall_score(y_test, y_pred_custom))

    print("Acurácia (default):", accuracy_score(y_test, y_pred_default))
    print("F1-score (default):", f1_score(y_test, y_pred_default))
    print("Recall (default):", recall_score(y_test, y_pred_default))

"""
[RESULTADOS COM FUNÇÃO DUMMY]
Acurácia (custom): 0.631578947368421
F1-score (custom): 0.7741935483870968
Recall (custom): 1.0
Acurácia (default): 0.8947368421052632
F1-score (default): 0.911764705882353
Recall (default): 0.8611111111111112
"""