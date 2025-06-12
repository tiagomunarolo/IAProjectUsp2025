import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "src/data/UCI_Credit_Card.csv")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# Importações do projeto
from src.data_processing.data_processing import DataProcessing
from src.tree.tree import DecisionTreeAdapted
from src.metrics.metrics import f1_score

# Critérios disponíveis
CRITERIA = [
    'gini',
    'entropy',
    'f1_proxy_impurity',
    'f1_gini',
    'f1_weighted_gini',
    'f1_split_impurity',
]


def split_data(x: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 42):
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    split_point = int(len(x) * (1 - test_size))
    return x[indices[:split_point]], x[indices[split_point:]], y[indices[:split_point]], y[indices[split_point:]]


def objective_function(x_train, x_val, y_train, y_val, max_depth, min_samples_split, criterion):
    tree = DecisionTreeAdapted(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
    )
    tree.fit(pd.DataFrame(x_train), pd.Series(y_train))
    y_pred = tree.predict(pd.DataFrame(x_val))

    f1 = f1_score(y_val, y_pred)
    return f1


def run_smbo(x: np.ndarray, y: np.ndarray, n_iter: int = 30, seed: int = 42):
    random.seed(seed)
    results = []

    x_train, x_val, y_train, y_val = split_data(x, y, test_size=0.3, seed=seed)

    for i in range(n_iter):
        max_depth = random.randint(3, 10)
        min_samples_split = random.choice([5, 10, 20, 30, 40, 50])
        criterion = random.choice(CRITERIA)

        try:
            f1 = objective_function(
                x_train, x_val, y_train, y_val,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
            )
        except Exception as e:
            print(f"[Iter {i+1}] ERRO: {e}")
            continue

        config = {
            'iteration': i,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'criterion': criterion,
            'f1_score': f1
        }

        results.append(config)
        print(f"[Iter {i+1:02d}/{n_iter}] ✅ F1: {f1:.4f} | Critério: {criterion}")

    return pd.DataFrame(results).sort_values(by='f1_score', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Carregando dados...")
    dp = DataProcessing(DATA_PATH)
    dp.process_dataset()
    x, y = dp.x.to_numpy(), dp.y.to_numpy()

    print("Iniciando SMBO (foco: F1 Score)...")
    df_resultados = run_smbo(x, y, n_iter=30, seed=RANDOM_STATE)

    print("\nTop resultados (por F1 Score):")
    print(df_resultados.head())

    df_resultados.to_csv("resultados_smbo_f1.csv", index=False)
