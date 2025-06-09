import numpy as np


def custom_cost_impurity(**kwargs):
    """
    Critério de impureza baseado no custo esperado de erros de classificação.
    Penaliza mais os falsos positivos (inadimplente previsto como adimplente),
    refletindo o contexto real de crédito.

    Argumentos esperados via kwargs:
        - y_left: array com os rótulos da divisão à esquerda
        - y_right: array com os rótulos da divisão à direita

    Parâmetros:
        - cost_fp: custo de um falso positivo (default=5.0)
        - cost_fn: custo de um falso negativo (default=1.0)

    Retorna:
        - custo total esperado ponderado pelo tamanho dos grupos
    """
    y_left = kwargs['y_left']
    y_right = kwargs['y_right']
    cost_fp = kwargs.get('cost_fp', 10.0)
    cost_fn = kwargs.get('cost_fn', 1.0)

    def group_cost(y):
        total = len(y)
        if total == 0:
            return 0.0
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        p_pos = n_pos / total
        p_neg = n_neg / total

        return (p_neg * cost_fn) + (p_pos * cost_fp)

    total_left = len(y_left)
    total_right = len(y_right)
    total = total_left + total_right

    cost_left = group_cost(y_left)
    cost_right = group_cost(y_right)

    total_cost = (total_left * cost_left + total_right * cost_right) / total
    return total_cost