from typing import Generator

import pandas as pd
from numpy.dtypes import BoolDType
from src.plot import plot_histogram
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold

# There are 25 variables:
#
# ID: ID of each client
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# SEX: Gender (1=male, 2=female)
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# AGE: Age in years
# PAY_0: Repayment status in September, 2005
#  -1=pay duly,
#   1=payment delay for one month,
#   2=payment delay for two months, …
#   8=payment delay for eight months,
#   9=payment delay for nine months and above
# PAY_2: Repayment status in August, 2005 (scale same as above)
# PAY_3: Repayment status in July, 2005 (scale same as above)
# PAY_4: Repayment status in June, 2005 (scale same as above)
# PAY_5: Repayment status in May, 2005 (scale same as above)
# PAY_6: Repayment status in April, 2005 (scale same as above)
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# default.payment.next.month: Default payment (1=yes, 0=no)

DUMMY_COLS = [
    'SEXO',
    'ESCOLARIDADE',
    'ESTADO_CIVIL',
]

RENAME_COLS = {
    'LIMIT_BAL': 'LIMITE_DE_CREDITO',
    'SEX': 'SEXO',
    'EDUCATION': 'ESCOLARIDADE',
    'MARRIAGE': 'ESTADO_CIVIL',
    'AGE': 'IDADE',
    'PAY_0': 'ATRASO_SETEMBRO',
    'PAY_2': 'ATRASO_AGOSTO',
    'PAY_3': 'ATRASO_JULHO',
    'PAY_4': 'ATRASO_JUNHO',
    'PAY_5': 'ATRASO_MAIO',
    'PAY_6': 'ATRASO_ABRIL',
    'BILL_AMT1': 'BOLETO_SETEMBRO',
    'BILL_AMT2': 'BOLETO_AGOSTO',
    'BILL_AMT3': 'BOLETO_JULHO',
    'BILL_AMT4': 'BOLETO_JUNHO',
    'BILL_AMT5': 'BOLETO_MAIO',
    'BILL_AMT6': 'BOLETO_ABRIL',
    'PAY_AMT1': 'VALOR_PAGO_SETEMBRO',
    'PAY_AMT2': 'VALOR_PAGO_AGOSTO',
    'PAY_AMT3': 'VALOR_PAGO_JULHO',
    'PAY_AMT4': 'VALOR_PAGO_JUNHO',
    'PAY_AMT5': 'VALOR_PAGO_MAIO',
    'PAY_AMT6': 'VALOR_PAGO_ABRIL',
    'default.payment.next.month': 'LABEL', }


class DataProcessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.x = None
        self.y = None

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    @staticmethod
    def rename_and_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
        # rename columns
        df.rename(columns=RENAME_COLS, inplace=True)
        # Drop ID
        df = df.drop('ID', axis=1)
        # Ajusta escolaridade
        df.ESCOLARIDADE = df.ESCOLARIDADE.replace({0: 4, 5: 4, 6: 4})
        # Estado civil (0 e 3 são tratados como solteiros)
        df.ESTADO_CIVIL = df.ESTADO_CIVIL.replace({0: 2, 3: 2})
        # 1=male, 2=female
        df.SEXO = df.SEXO.map({1: 'Masculino', 2: 'Feminino'})
        # 1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown
        df.ESCOLARIDADE = df.ESCOLARIDADE.map(
            {1: 'Pos_graduacao', 2: 'Graduacao', 3: 'Ensino_medio', 4: 'Outros'})
        # (1=married, 2=single, 3=others)
        df.ESTADO_CIVIL = df.ESTADO_CIVIL.map({1: 'Casado', 2: 'Solteiro', 3: 'Outros'})
        return df

    @staticmethod
    def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if isinstance(df[col].dtype, BoolDType):
                continue
            df[col] = df[col].astype('float64')
            df.loc[:, col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    def get_data(self, folds: int = 5) -> Generator:
        k_fold = KFold(n_splits=folds, shuffle=True, random_state=42)
        for train_index, test_index in k_fold.split(self.x):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            yield x_train, x_test, y_train, y_test

    @staticmethod
    def plot_dataset(df: pd.DataFrame) -> None:
        for col in df.columns:
            plot_histogram(df[col], title=col)

    def process_dataset(self, select_k: int = 0, plot_dataset: bool = False) -> None:
        df = self.load_dataset()
        df = self.rename_and_filter_columns(df)
        y = df.LABEL  # label
        if plot_dataset:
            self.plot_dataset(df=df)
        for col in DUMMY_COLS:
            df = pd.get_dummies(df, columns=[col])

        x = df.drop('LABEL', axis=1)
        if select_k:  # feature selection
            selector = SelectKBest(f_classif, k=select_k)
            selector.fit_transform(x, y)
            x = df[selector.get_feature_names_out()]

        # save df
        self.df = df
        # set type to float
        x = self.normalize_dataset(x)
        x = x.astype('float64')
        self.x = x
        y = y.astype('int32')
        self.y = y
