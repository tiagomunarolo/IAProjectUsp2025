import pandas as pd
from loguru import logger
from src.data_processing import DataProcessing
from sklearn.metrics import accuracy_score, f1_score
import os
from dotenv import load_dotenv

from src.tree import DecisionTreeAdapted

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MAX_DEPTH = int(os.getenv('MAX_DEPTH'))
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT'))
CRITERION = os.getenv('CRITERION', 'custom')


def process_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Processa o dataset e retorna os conjuntos de treino e teste """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    logger.info('Dataset loaded and pre-processed')
    return data_processing.process_dataset()


def train(tree: DecisionTreeAdapted) -> None:
    """ Treina a árvore de decisão """
    x_train, x_test, y_train, y_test = process_dataset()
    tree.fit(x_train, y_train)
    y_hat = tree.predict(x_test)
    logger.info(f'Accuracy: {accuracy_score(y_test, y_hat)}')
    logger.info(f'F1-Score: {f1_score(y_test, y_hat)}')


def main():
    logger.info('Training Decision Tree')
    tree = DecisionTreeAdapted(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        criterion=CRITERION
    )
    train(tree=tree)
    logger.info(tree.tree_)


if __name__ == '__main__':
    main()
