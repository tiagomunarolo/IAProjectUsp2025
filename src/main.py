import os
import numpy as np
from loguru import logger
from typing import Generator
from src.data_processing import DataProcessing
from src.tree.metrics.metrics import accuracy, f1_score
from dotenv import load_dotenv

from src.tree import DecisionTreeAdapted

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')  # Caminho para o dataset
MAX_DEPTH = int(os.getenv('MAX_DEPTH', 5))  # Profundidade máxima da árvore
FOLDS = max(2, int(os.getenv('FOLDS', 5)))  # Quantidade de folds na validação cruzada
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT', 2))  # Número mínimo de amostras necessárias para dividir um nó
CRITERION = os.getenv('CRITERION', 'f1_gini')  # Critério de divisão
HYBRID_MODEL = bool(os.getenv('HYBRID_MODEL', '') == 'True')  # Usar o modelo híbrido?

logger.info('Training Decision Tree')
logger.info(f'Using criterion: {CRITERION}')
logger.info(f'Using max_depth: {MAX_DEPTH}')
logger.info(f'Using min_samples_split: {MIN_SAMPLES_SPLIT}')
logger.info(f'Using hybrid_model: {HYBRID_MODEL}')


def process_dataset() -> Generator:
    """ Processa o dataset e retorna os conjuntos de treino e teste """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    data_processing.process_dataset()
    logger.info('Dataset loaded and pre-processed')
    return data_processing.get_data(folds=FOLDS)


def train(tree: DecisionTreeAdapted) -> None:
    """ Treina a árvore de decisão """
    accuracy_list, f1_list = [], []
    for x_train, x_test, y_train, y_test in process_dataset():
        tree.fit(x_train, y_train, hybrid=HYBRID_MODEL)
        y_hat = tree.predict(x_test)
        accuracy_list.append(accuracy(y_test, y_hat))
        f1_list.append(f1_score(y_test, y_hat))

    logger.info(f'Accuracy[{FOLDS}]: {np.mean(accuracy_list)}')
    logger.info(f'F1-Score[{FOLDS}]: {np.mean(f1_list)}')


def main():
    tree = DecisionTreeAdapted(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        criterion=CRITERION
    )
    train(tree=tree)
    logger.info(tree.tree_)


if __name__ == '__main__':
    main()
