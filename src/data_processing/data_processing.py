import pandas as pd
from sklearn.model_selection import train_test_split

DUMMY_COLS = ['SEX',
              'EDUCATION',
              'MARRIAGE',
              'PAY_0',
              'PAY_2',
              'PAY_3',
              'PAY_4',
              'PAY_5',
              'PAY_6']


class DataProcessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    def process_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_dataset()
        columns = df.columns.tolist()
        x = df[columns[-3:-1]]  # features
        y = df[columns[-1]]  # label
        for col in DUMMY_COLS:
            if col not in x.columns:
                continue
            x = pd.get_dummies(x, columns=[col])
        # treino e teste (90/10)
        return train_test_split(x, y,
                                stratify=y,  # estratifica o dataset
                                shuffle=True,  # embaralha
                                test_size=0.1,  # 10% para teste
                                random_state=42)
