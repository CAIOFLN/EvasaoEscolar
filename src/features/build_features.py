"""Engenharia de atributos.

Decisões validadas por validação cruzada (5-fold, F1 macro), não só pela EDA:

- TODAS AS FEATURES SÃO MANTIDAS. As variáveis macroeconômicas/demográficas têm
  correlação linear ~0 com o alvo, mas removê-las PIOROU o F1 na CV — o XGBoost
  extrai sinal não-linear delas. Por isso não há descarte de colunas.
- CRIAÇÃO: o ganho real vem de **razões** (ex.: aprovados/matriculados). Árvores
  de decisão não conseguem derivar divisões internamente, então essas features
  agregam informação. Já somas e diferenças (totais, reprovações = matriculados −
  aprovados) o XGBoost reconstrói sozinho — testadas, foram redundantes/ruído e
  pioraram o modelo, por isso não entram.
- CATEGÓRICAS: nominais codificadas como inteiros E binárias (0/1) são marcadas
  como `category`, para o XGBoost as tratar de forma nativa (e não como números
  ordenados). Semanticamente são rótulos, não quantidades.
"""
import numpy as np
import pandas as pd

from src.data.load import load_raw

# Categorias nominais (sem ordem) codificadas como inteiros. Marcadas como
# `category` para o XGBoost não assumir ordem entre os códigos.
NOMINAL_FEATURES = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Variáveis binárias (0/1): semanticamente são categóricas (sim/não), não
# números. Para árvores o corte é equivalente, mas tratá-las como `category`
# é mais correto e deixa a intenção explícita.
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Todas as colunas tratadas como categóricas
CATEGORICAL_FEATURES = NOMINAL_FEATURES + BINARY_FEATURES


def _safe_ratio(num, den):
    """Razão elemento a elemento, retornando 0 onde o denominador é 0."""
    return np.where(den > 0, num / den, 0.0)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as razões de aprovação que comprovadamente melhoram o modelo.

    Apenas duas features, selecionadas por ablação com validação cruzada:
    - overall_approval_rate: aprovados / matriculados nos dois semestres juntos.
    - approval_rate_2nd: aprovados / matriculados no 2º semestre (o mais
      discriminante na EDA).

    Versões com mais razões, agregados, reprovações, engajamento e trajetória
    foram testadas e não melhoraram (ver docstring do módulo).
    """
    df = df.copy()

    e1 = df["Curricular units 1st sem (enrolled)"]
    a1 = df["Curricular units 1st sem (approved)"]
    e2 = df["Curricular units 2nd sem (enrolled)"]
    a2 = df["Curricular units 2nd sem (approved)"]

    df["approval_rate_2nd"] = _safe_ratio(a2, e2)
    df["overall_approval_rate"] = _safe_ratio(a1 + a2, e1 + e2)

    return df


def prepare(df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
    """Aplica a engenharia de atributos e tipa as nominais.

    Mantém todas as colunas originais. Retorna o DataFrame pronto para separar
    X/y, com as nominais como `category` (compatível com `enable_categorical=True`
    do XGBoost).
    """
    out = df.copy()

    if add_features:
        out = engineer(out)

    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


def main():
    df = prepare(load_raw())
    print(df.info())


if __name__ == "__main__":
    main()
