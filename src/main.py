import os
import numpy as np
from loguru import logger
from typing import Generator
from src.data_processing import DataProcessing
from src.metrics.metrics import accuracy, f1_score
from dotenv import load_dotenv

from src.tree import DecisionTreeAdapted

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')  # Caminho para o dataset
MAX_DEPTH = int(os.getenv('MAX_DEPTH', 5))  # Profundidade máxima da árvore
FOLDS = max(2, int(os.getenv('FOLDS', 5)))  # Quantidade de folds na validação cruzada
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT', 2))  # Número mínimo de amostras necessárias para dividir um nó
CRITERION = os.getenv('CRITERION', 'f1_gini')  # Critério de divisão
SELECT_K = int(os.getenv('SELECT_K', 0))  # Quantidade de features/colunas a serem selecionadas
PLOT_DATASET = bool(os.getenv('PLOT_DATASET', '') == 'True')  # Plotar o dataset
HYBRID_MODEL = bool(os.getenv('HYBRID_MODEL', '') == 'True')  # Usar o modelo híbrido?

logger.info('Treinando: Decision Tree')
logger.info(f'Usando Critério: {CRITERION}')
logger.info(f'Usando Profundidade: {MAX_DEPTH}')
logger.info(f'Usando Número Minimo de Amostras por Nó: {MIN_SAMPLES_SPLIT}')
logger.info(f'Usando Quantidade de Folds: {FOLDS}')
logger.info(f'Usando Quantidade de Features Selecionadas? {SELECT_K if SELECT_K > 0 else "Não"}')
logger.info(f'Usando Modelo Híbrido: {HYBRID_MODEL}')


def process_dataset() -> Generator:
    """ Processa o dataset e retorna os conjuntos de treino e teste """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    data_processing = DataProcessing(dataset_path=DATA_PATH)
    data_processing.process_dataset(select_k=SELECT_K,
                                    plot_dataset=PLOT_DATASET)
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

    logger.info(f'Accuracy[{FOLDS}-Folds]: {np.mean(accuracy_list)}')
    logger.info(f'F1-Score[{FOLDS}-Folds]: {np.mean(f1_list)}')


def main():
    """
    Inicializa o algoritmo
    Treina a árvore de decisão
    """
    tree = DecisionTreeAdapted(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        criterion=CRITERION
    )
    train(tree=tree)
    logger.info(tree.tree_)


if __name__ == '__main__':
    """
    Main function
    """
    main()
