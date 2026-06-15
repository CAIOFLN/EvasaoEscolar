"""Baixa o dataset "Predict Students' Dropout and Academic Success" (UCI id=697).

Uso:
    python scripts/download_data.py

Salva o CSV em data/raw/dataset.csv. Esse diretório é ignorado pelo git,
então cada pessoa da dupla roda este script localmente uma vez.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT = RAW_DIR / "dataset.csv"
DATASET_ID = 697


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Baixando dataset UCI id={DATASET_ID}...")
    ds = fetch_ucirepo(id=DATASET_ID)

    # Junta features (X) e alvo (y) em um único DataFrame.
    df = ds.data.features.copy()
    target_col = ds.data.targets.columns[0]
    df[target_col] = ds.data.targets[target_col]

    df.to_csv(OUTPUT, index=False)
    print(f"Salvo em {OUTPUT}  ({df.shape[0]} linhas, {df.shape[1]} colunas)")
    print(f"Coluna alvo: '{target_col}' -> classes: {sorted(df[target_col].unique())}")


if __name__ == "__main__":
    main()
