import pandas as pd
from loguru import logger
from src.data_processing import DataProcessing
from sklearn.metrics import accuracy_score
from src.tree import DecisionTree
from pathlib import Path

DATA_PATH = Path(__file__).parent.joinpath('data/UCI_Credit_Card.csv')


def process_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    logger.info('Dataset loaded and pre-processed')
    return data_processing.process_dataset()


def train() -> DecisionTree:
    x_train, x_test, y_train, y_test = process_dataset()
    tree = DecisionTree(max_depth=3)
    tree.fit(x_train, y_train)
    y_hat = tree.predict(x_test)
    logger.info(f'Accuracy: {accuracy_score(y_test, y_hat)}')
    return tree


def main():
    logger.info('Training...')
    tree = train()
    logger.info(tree.tree_)


if __name__ == '__main__':
    main()
