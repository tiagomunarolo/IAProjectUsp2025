import numpy as np

def custom_cost_impurity(**kwargs):
    y_left, y_right = kwargs['y_left'], kwargs['y_right']

    def calcula_custo(y):
        if len(np.unique(y)) == 1:
            return 0, 0
        # Classe 1 = inadimplente
        majority = np.bincount(y).argmax()
        predicted = np.full(len(y), majority)
        fp = np.sum((predicted == 1) & (y == 0))
        fn = np.sum((predicted == 0) & (y == 1))
        return fp, fn

    fp_left, fn_left = calcula_custo(y_left)
    fp_right, fn_right = calcula_custo(y_right)

    total = len(y_left) + len(y_right)
    FP_PESO = 1.0
    FN_PESO = 5.0

    custo_total = (fp_left + fp_right) * FP_PESO + (fn_left + fn_right) * FN_PESO
    return custo_total / total if total > 0 else np.inf
