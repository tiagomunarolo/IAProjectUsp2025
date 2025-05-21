# https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download
import os
from loguru import logger
from src.data_processing import DataProcessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from dotenv import load_dotenv

import pandas as pd

"""
Arquivo de testes
Rodar o algoritmo AS-IS com o dataset e comparar com o nosso
"""
load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MAX_DEPTH = int(os.getenv('MAX_DEPTH'))
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT'))


def process_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    logger.info('Dataset loaded and pre-processed')
    return data_processing.process_dataset()


def train():
    x_train, x_test, y_train, y_test = process_dataset()
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_SPLIT)
    tree.fit(x_train, y_train)
    y_hat = tree.predict(x_test)
    logger.info(f'Accuracy: {accuracy_score(y_test, y_hat)}')
    return tree


def main():
    logger.debug('[TEST_ONLY]Training...')
    train()


if __name__ == '__main__':
    main()
