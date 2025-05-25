import os
import pandas as pd
from dotenv import load_dotenv
from src.data_processing import DataProcessing
from src.experiments.smbo_manual import smbo_busca

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FOLDS = int(os.getenv("FOLDS", 5))
CRITERION = os.getenv("CRITERION", "f1_gini")
N_ITER = int(os.getenv("SMBO_ITER", 15))
HYBRID = os.getenv("HYBRID_MODEL", "False") == "True"

def main():
    print(f"Critério: {CRITERION} | Hybrid: {HYBRID} | Iterações: {N_ITER}")

    dp = DataProcessing(DATA_PATH)
    dp.process_dataset()
    folds = list(dp.get_data(folds=FOLDS))

    resultados = smbo_busca(
        folds=folds,
        criterion=CRITERION,
        hybrid=HYBRID,
        n_iter=N_ITER
    )

    df = pd.DataFrame(resultados)
    os.makedirs("resultados", exist_ok=True)
    df.to_csv(f"resultados/smbo_{CRITERION}.csv", index=False)
    print("Finalizado. Resultados salvos em 'resultados/'")

if __name__ == "__main__":
    main()
