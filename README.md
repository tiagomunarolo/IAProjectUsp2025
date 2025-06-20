# IAProjectUsp2025

Arvore de decisão com impureza customizada

# 1. Considerações importantes para execução do projeto:

## Python instalado
`python --version`

## Instalar dependências:
`pip install -r requirements.txt`

## Build dos arquivos em cython:
`python setup.py build_ext --inplace`

## Adicionar a raiz do projeto ao path do python (Exemplo abaixo):
`export PYTHONPATH=$PYTHONPATH:/Users/tiago_m2/PycharmProjects/IAProjectUsp2025`

# 2. Criar .env:
Na raiz do projeto: criar um arquivo .env com as seguintes variáveis(Exemplo):

```bash 
DATA_PATH=/Users/tiago_m2/PycharmProjects/IAProjectUsp2025/src/data/UCI_Credit_Card.csv
MAX_DEPTH=5
MIN_SAMPLES_SPLIT=50
HYBRID_MODEL=None
SELECT_K=0
FOLDS=5
CRITERION=f1_weighted_gini
PLOT_DATASET=False
```

# 3. Como executar o projeto:
```bash
  python src/main.py
```

Estrutura do Diretório:
src
├── main.py (main file)
├── data_processing
│ ├── data_processing.py (data processing)
├── data
│ ├── CSV dataset
├── testing
│ ├── Testing folder only
├── tree
│ ├── interfaces
│ │ ├── tree.py
│ │
│ ├── tree.py (implementation)