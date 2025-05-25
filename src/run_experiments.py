import os
import time
import pandas as pd
from src.data_processing import DataProcessing
from src.tree import DecisionTreeAdapted
from sklearn.metrics import accuracy_score, f1_score, recall_score
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FOLDS = int(os.getenv("FOLDS", 5))

CRITERIA = [
    'gini',
    'f1_proxy_impurity',
    'f1_gini',
    'logistic_regression_impurity',
    'custom_cost_impurity' 
]

DEPTHS = [3, 4]
MIN_SAMPLES = [10, 20]
HYBRIDS = [False, True]

def process_dataset():
    dp = DataProcessing(DATA_PATH)
    dp.process_dataset()
    return list(dp.get_data(folds=FOLDS))

results = []

for criterion in CRITERIA:
    for depth in DEPTHS:
        for min_samples in MIN_SAMPLES:
            for hybrid in HYBRIDS:
                accs, f1s, recalls = [], [], []
                folds = process_dataset()
                start = time.time()

                for x_train, x_test, y_train, y_test in folds:
                    tree = DecisionTreeAdapted(
                        max_depth=depth,
                        min_samples_split=min_samples,
                        criterion=criterion
                    )
                    tree.fit(x_train, y_train, hybrid=hybrid)
                    y_pred = tree.predict(x_test)

                    accs.append(accuracy_score(y_test, y_pred))
                    f1s.append(f1_score(y_test, y_pred))
                    recalls.append(recall_score(y_test, y_pred, pos_label=1))

                end = time.time()

                results.append({
                    'criterion': criterion,
                    'max_depth': depth,
                    'min_samples_split': min_samples,
                    'hybrid': hybrid,
                    'accuracy': sum(accs) / len(accs),
                    'f1_score': sum(f1s) / len(f1s),
                    'recall_1': sum(recalls) / len(recalls),
                    'exec_time_sec': round(end - start, 2)
                })

                print(f' {criterion} | depth={depth}, min_split={min_samples}, hybrid={hybrid}')

df = pd.DataFrame(results)
os.makedirs('resultados', exist_ok=True)
df.to_csv('resultados/experimentos_comparativos.csv', index=False)
print("Experimentos finalizados. Resultados salvos em: resultados/experimentos_comparativos.csv")
