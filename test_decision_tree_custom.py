# ==========================================
# Custom Decision Tree com Critério Alternativo
# ==========================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

MAX_DEPTH = 2

# ----------------------
# Critérios de impureza
# ----------------------
def gini_impurity(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def cost_sensitive_impurity(y_true, y_pred, cost_fn=5, cost_fp=1):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return cost_fn * FN + cost_fp * FP

def surrogate_logreg_impurity(X, y):
    if len(np.unique(y)) == 1:
        return 0
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return 1 - model.score(X, y)  # quanto menor a acurácia, maior a impureza

# ----------------------
# Classe da Árvore Customizada
# ----------------------
class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, impurity_fn=surrogate_logreg_impurity):
        self.max_depth = max_depth
        self.tree_ = None
        self.impurity_fn = impurity_fn

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        X = SelectKBest(chi2, k=min(10, X.shape[1])).fit_transform(X, y)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _custom_criterion(self, y_left, x_left, y_right, x_right):
        return self.impurity_fn(x_left, y_left) + self.impurity_fn(x_right, y_right)

    def _best_split(self, X, y):
        m, n = X.shape
        best_feature, best_thresh = None, None
        best_criterion = float('inf')

        for index, feature in enumerate(range(n)):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 100:
                thresholds = np.linspace(X[:, feature].min(), X[:, feature].max(), 100)

            for index_t, t in enumerate(thresholds):
                y_left = y[X[:, feature] <= t]
                y_right = y[X[:, feature] > t]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                x_left = X[X[:, feature] <= t]
                x_right = X[X[:, feature] > t]
                score = self._custom_criterion(y_left, x_left, y_right, x_right)

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

# ----------------------
# Execução de teste
# ----------------------
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    tree = CustomDecisionTreeClassifier(max_depth=MAX_DEPTH)
    default_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini')

    default_tree.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    y_pred_custom = tree.predict(X_test)
    y_pred_default = default_tree.predict(X_test)

    print("\n[RESULTADOS]")
    print("Acurácia (custom):", accuracy_score(y_test, y_pred_custom))
    print("F1-score (custom):", f1_score(y_test, y_pred_custom))
    print("Recall (custom):", recall_score(y_test, y_pred_custom))

    print("Acurácia (default):", accuracy_score(y_test, y_pred_default))
    print("F1-score (default):", f1_score(y_test, y_pred_default))
    print("Recall (default):", recall_score(y_test, y_pred_default))
