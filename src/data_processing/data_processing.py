import pandas as pd
from numpy.dtypes import BoolDType
from sklearn.model_selection import train_test_split

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
    'ATRASO_PAGTO_SETEMBRO',
    'ATRASO_PAGTO_AGOSTO',
    'ATRASO_PAGTO_JULHO',
    'ATRASO_PAGTO_JUNHO',
    'ATRASO_PAGTO_MAIO',
    'ATRASO_PAGTO_ABRIL',
]


class DataProcessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    @staticmethod
    def rename_and_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns={
            'LIMIT_BAL': 'LIMITE_DE_CREDITO',
            'SEX': 'SEXO',
            'EDUCATION': 'ESCOLARIDADE',
            'MARRIAGE': 'ESTADO_CIVIL',
            'AGE': 'IDADE',
            'PAY_0': 'ATRASO_PAGTO_SETEMBRO',
            'PAY_2': 'ATRASO_PAGTO_AGOSTO',
            'PAY_3': 'ATRASO_PAGTO_JULHO',
            'PAY_4': 'ATRASO_PAGTO_JUNHO',
            'PAY_5': 'ATRASO_PAGTO_MAIO',
            'PAY_6': 'ATRASO_PAGTO_ABRIL',
            'BILL_AMT1': 'VALOR_BOLETO_SETEMBRO',
            'BILL_AMT2': 'VALOR_BOLETO_AGOSTO',
            'BILL_AMT3': 'VALOR_BOLETO_JULHO',
            'BILL_AMT4': 'VALOR_BOLETO_JUNHO',
            'BILL_AMT5': 'VALOR_BOLETO_MAIO',
            'BILL_AMT6': 'VALOR_BOLETO_ABRIL',
            'PAY_AMT1': 'VALOR_PAGO_ATE_SETEMBRO',
            'PAY_AMT2': 'VALOR_PAGO_ATE_AGOSTO',
            'PAY_AMT3': 'VALOR_PAGO_ATE_JULHO',
            'PAY_AMT4': 'VALOR_PAGO_ATE_JUNHO',
            'PAY_AMT5': 'VALOR_PAGO_ATE_MAIO',
            'PAY_AMT6': 'VALOR_PAGO_ATE_ABRIL',
            'default.payment.next.month': 'LABEL',

        }, inplace=True)
        df = df.drop('ID', axis=1)
        # 1=male, 2=female
        df.SEXO = df.SEXO.map({1: 'Masculino', 2: 'Feminino'})
        # 1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown
        df.ESCOLARIDADE = df.ESCOLARIDADE.map(
            {1: 'Pos_graduacao', 2: 'Graduacao', 3: 'Ensino_medio', 4: 'Outros', 5: 'Desconhecido', 6: 'Desconhecido'})
        # (1=married, 2=single, 3=others)
        df.ESTADO_CIVIL = df.ESTADO_CIVIL.map({1: 'Casado', 2: 'Solteiro', 3: 'Outros'})
        # -1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ...
        # < 0 = antecipado, 0 = nao_atrasado, > 0 = atrasado
        for col in ['ATRASO_PAGTO_SETEMBRO', 'ATRASO_PAGTO_AGOSTO', 'ATRASO_PAGTO_JULHO', 'ATRASO_PAGTO_JUNHO',
                    'ATRASO_PAGTO_MAIO', 'ATRASO_PAGTO_ABRIL']:
            df[col] = df[col].apply(lambda x: 'atrasado' if x > 0 else x)
            df[col] = df[col].apply(lambda x: 'nao_atrasado' if not x else x)
            df[col] = df[col].apply(lambda x: 'antecipado' if isinstance(x, int) and x < 0 else x)

        return df

    @staticmethod
    def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if isinstance(df[col].dtype, BoolDType):
                continue
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df = df.astype('float32')
        return df

    def process_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_dataset()
        df = self.rename_and_filter_columns(df)
        x = df.drop('LABEL', axis=1)
        y = df.LABEL  # label
        for col in DUMMY_COLS:
            x = pd.get_dummies(x, columns=[col])

        x = self.normalize_dataset(x)
        # treino e teste (90/10)
        return train_test_split(x, y,
                                stratify=y,  # estratifica o dataset
                                shuffle=True,  # embaralha
                                test_size=0.1,  # 10% para teste
                                random_state=42)
