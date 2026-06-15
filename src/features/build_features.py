"""Engenharia e seleção de atributos.

Decisões guiadas pela análise exploratória (notebooks/01_eda.ipynb):

- REMOÇÃO: variáveis com correlação ~0 com o alvo (macroeconômicas e algumas
  demográficas) — pouco valor preditivo e ruído potencial.
- CRIAÇÃO: features derivadas do desempenho acadêmico, que foi o bloco mais
  discriminante (taxas de aprovação, agregados e evolução das notas).
- NOMINAIS: categorias codificadas como inteiros são marcadas como `category`
  para que o XGBoost as trate de forma nativa (e não como números ordenados).
"""
import numpy as np
import pandas as pd

# Variáveis descartadas (correlação ~0 com o alvo na EDA)
DROP_FEATURES = [
    "Nacionality",
    "International",
    "Educational special needs",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Categorias nominais (sem ordem) codificadas como inteiros
NOMINAL_FEATURES = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona atributos derivados do desempenho acadêmico."""
    df = df.copy()

    e1 = df["Curricular units 1st sem (enrolled)"]
    a1 = df["Curricular units 1st sem (approved)"]
    e2 = df["Curricular units 2nd sem (enrolled)"]
    a2 = df["Curricular units 2nd sem (approved)"]
    g1 = df["Curricular units 1st sem (grade)"]
    g2 = df["Curricular units 2nd sem (grade)"]

    # Taxas de aprovação (0 quando não houve matrícula em disciplinas)
    df["approval_rate_1st"] = np.where(e1 > 0, a1 / e1, 0.0)
    df["approval_rate_2nd"] = np.where(e2 > 0, a2 / e2, 0.0)

    # Agregados dos dois semestres
    df["total_approved"] = a1 + a2
    df["total_enrolled"] = e1 + e2
    df["overall_approval_rate"] = np.where(
        df["total_enrolled"] > 0, df["total_approved"] / df["total_enrolled"], 0.0
    )

    # Notas: média e evolução entre semestres
    df["avg_grade"] = (g1 + g2) / 2
    df["grade_trend"] = g2 - g1

    return df


def prepare(
    df: pd.DataFrame,
    target: str = "Target",
    drop: bool = True,
    add_features: bool = True,
) -> pd.DataFrame:
    """Aplica seleção + engenharia de atributos e tipa as nominais.

    Retorna o DataFrame pronto para separar X/y. As nominais ficam como
    `category` (compatível com `enable_categorical=True` do XGBoost).
    """
    out = df.copy()

    if add_features:
        out = engineer(out)

    if drop:
        out = out.drop(columns=[c for c in DROP_FEATURES if c in out.columns])

    for col in NOMINAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out
