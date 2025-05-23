from .gini import gini_impurity as gini
from loguru import logger
from typing import Callable


class Criterion:
    @staticmethod
    def get_criterion(criterion: str) -> Callable:
        try:
            # importa dinamicamente
            exec(f'from src.tree.criterion import {criterion}')
            criteria = eval(criterion)
            if criteria and callable(criteria):
                # retorna a funcao
                return criteria
            raise ValueError
        except (ValueError, NameError, ImportError):
            logger.warning(f'Criterion {criterion} not found. Using default criterion (gini).')
            return gini
