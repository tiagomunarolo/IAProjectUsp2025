import itertools
import os
from loguru import logger
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data_processing.data_processing import DataProcessing
from src.tree import DecisionTreeAdapted
from src.tree.metrics.metrics import recall, f1_score

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
FOLDS = int(os.getenv("FOLDS", 5))

# Lista apenas com os nomes (strings) aceitos por criterion.py
CRITERIA = [
"gini", "entropy", "f1_gini", "f1_proxy_impurity", "custom_cost_impurity"
 ]

def get_data():
    data_proc = DataProcessing(DATA_PATH)
    data_proc.process_dataset()
    return data_proc.get_data(FOLDS)

def run():
    results = []
    max_depth_opts = [2, 3, 4, 5]
    min_samples_opts = [2, 5, 10, 20]

    for max_depth, min_samples, criterion_name in itertools.product(max_depth_opts, min_samples_opts, CRITERIA):
        recalls, f1s = [], []

        logger.info(f"Rodando config: depth={max_depth}, min_samples={min_samples}, criterion={criterion_name}")
        for x_train, x_test, y_train, y_test in get_data():
            model = DecisionTreeAdapted(
                max_depth=max_depth,
                min_samples_split=min_samples,
                criterion=criterion_name  # <- passa só o nome, conforme esperado em criterion.py
            )

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            recalls.append(recall(y_pred, y_test.to_numpy()))
            f1s.append(f1_score(y_pred, y_test.to_numpy()))

        results.append({
            "max_depth": max_depth,
            "min_samples_split": min_samples,
            "criterion": criterion_name,
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="recall", ascending=False)
    df_sorted.to_csv("smbo_results.csv", index=False)
    logger.success("SMBO finalizado. Resultados salvos em smbo_results.csv")

if __name__ == "__main__":
    run()
