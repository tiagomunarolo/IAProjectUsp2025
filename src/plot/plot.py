from pathlib import Path
import pandas as pd
import seaborn as sns

data_dir = Path(__file__).parent.parent.joinpath('data').resolve()


def plot_histogram(x: pd.Series, title: str):
    bins = min(100, len(x.unique()))
    plt = sns.displot(x, bins=bins, kde=True)
    plt.set(title=title)
    plt.set(ylabel='Contagem')
    plt.savefig(data_dir.joinpath(f'{title}.png'))
