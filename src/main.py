import os
import numpy as np
from loguru import logger
from typing import Generator
from src.data_processing import DataProcessing
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

from src.tree import DecisionTreeAdapted

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MAX_DEPTH = int(os.getenv('MAX_DEPTH', 5))
FOLDS = int(os.getenv('FOLDS', 10))
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT', 2))
CRITERION = os.getenv('CRITERION', 'f1_gini')

logger.info('Training Decision Tree')
logger.info(f'Using criterion: {CRITERION}')
logger.info(f'Using max_depth: {MAX_DEPTH}')
logger.info(f'Using min_samples_split: {MIN_SAMPLES_SPLIT}')


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
    accuracy, f1 = [], []
    for x_train, x_test, y_train, y_test in process_dataset():
        tree.fit(x_train, y_train)
        y_hat = tree.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_hat))
        f1.append(f1_score(y_test, y_hat))

    logger.info(f'Accuracy[{FOLDS}]: {np.mean(accuracy)}')
    logger.info(f'F1-Score[{FOLDS}]: {np.mean(f1)}')


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
