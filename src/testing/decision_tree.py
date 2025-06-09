# https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download
import os

import numpy as np
from typing import Generator
from loguru import logger
from src.data_processing import DataProcessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from dotenv import load_dotenv

"""
Arquivo de testes
Rodar o algoritmo AS-IS com o dataset e comparar com o nosso
"""
load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MAX_DEPTH = int(os.getenv('MAX_DEPTH'))
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT'))
FOLDS = int(os.getenv('FOLDS', 5))


def process_dataset() -> Generator:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    data_processing.process_dataset()
    logger.info('Dataset loaded and pre-processed')
    return data_processing.get_data(folds=FOLDS)


def test_train():
    accuracy, f1 = [], []
    for x_train, x_test, y_train, y_test in process_dataset():
        tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_SPLIT)
        tree.fit(x_train, y_train)
        y_hat = tree.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_hat))
        f1.append(f1_score(y_test, y_hat))

    logger.info(f'Accuracy[{FOLDS}]: {np.mean(accuracy)}')
    logger.info(f'F1-Score [{FOLDS}]: {np.mean(f1)}')


def main():
    logger.debug('[TEST_ONLY]Training...')
    test_train()


if __name__ == '__main__':
    main()
