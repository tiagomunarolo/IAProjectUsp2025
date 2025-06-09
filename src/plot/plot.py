from pathlib import Path
import pandas as pd
import seaborn as sns

data_dir = Path(__file__).parent.parent.joinpath('data').resolve()


def plot_histogram(x: pd.Series, title: str):
    if len(x.unique()) > 100:
        # Remove outliers
        # Remove a margem de 2.5% e 97.5%, facilitando a visualização
        x = x[(x >= x.quantile(0.025)) & (x <= x.quantile(0.975))]
    bins = min(100, len(x.unique()))
    plt = sns.displot(x, bins=bins, kde=True)
    plt.set(title=title)
    plt.set(ylabel='Contagem')
    plt.savefig(data_dir.joinpath(f'{title}.png'))
