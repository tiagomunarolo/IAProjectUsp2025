# Dataset 1:
# https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download

# COLS:
# ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default.payment.next.month']
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import preprocessing


def get_dataset() -> pd.DataFrame:
    return pd.read_csv('./UCI_Credit_Card.csv', sep=",")


def get_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_dataset()
    # remove variavel o ==> id
    # Y => classe a ser predita
    Y = df[list(df.columns)[-1]]
    X = df[list(df.columns)[1:-1]]
    for col in ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
        # transforma variáveis categoricas em numericas
        X = pd.get_dummies(X, columns=[col])
    # normaliza os dados
    # TODO: selecionar as melhores variáveis
    X = preprocessing.normalize(X)
    return train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=42)


def get_pipe_model() -> Pipeline:
    return Pipeline(steps=[
        ('normalizer', preprocessing.Normalizer()),
        ('tree', DecisionTreeClassifier())]
    )


def get_grid_classifier() -> GridSearchCV:
    # Tunando hiper parâmetros com busca de grid
    pipe_model = get_pipe_model()
    parameters = {'tree__max_depth': [3, 4, 5], 'tree__criterion': ['gini', 'entropy']}
    # Validação cruzada com 5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe_model, param_grid=parameters,
                        cv=cv, n_jobs=-1, verbose=1)
    return grid


def main():
    # Split do dataset
    X_train, X_test, y_train, y_test = get_train_test()
    # Instancia do classificador
    classifier = get_grid_classifier()
    # Treinamento
    classifier.fit(X_train, y_train)
    # Validaçao de acurácia
    score = accuracy_score(y_test, classifier.predict(X_test))
    print("Accuracy: ", score)
    print('Best estimator: ', classifier.best_estimator_)


if __name__ == '__main__':
    main()
