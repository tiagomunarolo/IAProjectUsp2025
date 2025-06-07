# IAProjectUsp2025

Arvore de decisão com impureza customizada

# Considerações importantes
Build dos arquivos em cython: `python setup.py build_ext --inplace`

# Criar .env:
Na raiz do projeto: criar um arquivo .env com as seguintes variáveis(Exemplo):

```bash 
DATA_PATH=/Users/tiago_m2/PycharmProjects/IAProjectUsp2025/src/data/UCI_Credit_Card.csv
MAX_DEPTH=5
MIN_SAMPLES_SPLIT=2
HYBRID_MODEL=False
SELECT_K=0
FOLDS=5
CRITERION=f1_gini
PLOT_DATASET=False
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