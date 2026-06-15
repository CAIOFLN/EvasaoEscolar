"""Carregamento e divisão dos dados."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "dataset.csv"

TARGET = "Target"  # classes: Dropout, Enrolled, Graduate


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Carrega o CSV bruto. Rode scripts/download_data.py antes, se não existir."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode: python scripts/download_data.py"
        )
    return pd.read_csv(path)


def split_X_y(df: pd.DataFrame, target: str = TARGET):
    """Separa features (X) do alvo (y)."""
    return df.drop(columns=[target]), df[target]


def train_test(df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2,
               random_state: int = 42):
    """Divisão treino/teste estratificada pelo alvo."""
    X, y = split_X_y(df, target)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
