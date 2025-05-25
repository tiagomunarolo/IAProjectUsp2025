import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
from src.tree import DecisionTreeAdapted

def avalia_configuracao(folds, max_depth, min_samples_split, criterion, hybrid):
    accs, f1s, recalls = [], [], []
    for x_train, x_test, y_train, y_test in folds:
        tree = DecisionTreeAdapted(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        )
        tree.fit(x_train, y_train, hybrid=hybrid)
        y_pred = tree.predict(x_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred, pos_label=1))

    return {
        'accuracy': np.mean(accs),
        'f1': np.mean(f1s),
        'recall_1': np.mean(recalls)
    }

def smbo_busca(folds, criterion='gini', hybrid=False, n_iter=10):
    historico = []
    avaliado = set()
    for _ in range(n_iter):
        while True:
            max_depth = random.choice([2, 3, 4, 5])
            min_samples_split = random.choice([5, 10, 15, 20])
            chave = (max_depth, min_samples_split)
            if chave not in avaliado:
                avaliado.add(chave)
                break

        resultado = avalia_configuracao(
            folds=folds,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            hybrid=hybrid
        )
        resultado['max_depth'] = max_depth
        resultado['min_samples_split'] = min_samples_split
        resultado['criterion'] = criterion
        resultado['hybrid'] = hybrid

        print(f"[{_+1:02d}/{n_iter}] depth={max_depth}, split={min_samples_split} "
              f"=> F1={resultado['f1']:.4f}, Recall_1={resultado['recall_1']:.4f}")

        historico.append(resultado)

    return sorted(historico, key=lambda x: x['f1'], reverse=True)