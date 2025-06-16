import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data_processing import DataProcessing
from src.metrics.metrics import f1_score
from src.tree import DecisionTreeAdapted
from src.tree.tree import HybridOption

# 1. Processar o dataset
DATA_PATH = "src/data/UCI_Credit_Card.csv"
dp = DataProcessing(dataset_path=DATA_PATH)
dp.process_dataset(select_k=0, plot_dataset=False)
folds = list(dp.get_data(folds=5))  # mesmo que você já usava

# 2. Função genérica de avaliação
def avaliar_modelo(modelo, folds, is_tree=False):
    f1_list = []
    for x_train, x_test, y_train, y_test in folds:
        modelo.fit(x_train, y_train)
        y_pred = modelo.predict(x_test) if is_tree else modelo.predict(x_test)
        f1_list.append(f1_score(y_pred, y_test))
    return f1_list

# 3. Modelo proposto (sua árvore)
tree = DecisionTreeAdapted(
    max_depth=15,
    min_samples_split=25,
    criterion='f1_gini',
    hybrid_model=HybridOption.NONE
)
f1_proposto = avaliar_modelo(tree, folds, is_tree=True)

# 4. Random Forest e XGBoost
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
f1_rf = avaliar_modelo(rf, folds)

xgb = XGBClassifier(n_estimators=50, max_depth=5, use_label_encoder=False,
                    eval_metric="logloss", random_state=42)
f1_xgb = avaliar_modelo(xgb, folds)

# 5. Testes estatísticos
t_rf, p_rf = ttest_rel(f1_proposto, f1_rf)
t_xgb, p_xgb = ttest_rel(f1_proposto, f1_xgb)
w_rf, p_w_rf = wilcoxon(f1_proposto, f1_rf)
w_xgb, p_w_xgb = wilcoxon(f1_proposto, f1_xgb)

# 6. Exibir resultados
print("F1-score médio por modelo:")
print(f"Proposto: {np.mean(f1_proposto):.4f}")
print(f"Random Forest: {np.mean(f1_rf):.4f}")
print(f"XGBoost: {np.mean(f1_xgb):.4f}\n")

print("Teste T pareado:")
print(f"RF: t={t_rf:.4f}, p={p_rf:.4f}")
print(f"XGB: t={t_xgb:.4f}, p={p_xgb:.4f}\n")

print("Teste de Wilcoxon:")
print(f"RF: p={p_w_rf:.4f}")
print(f"XGB: p={p_w_xgb:.4f}")
