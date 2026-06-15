"""Engenharia e seleção de atributos.

Decisões validadas por validação cruzada (5-fold, F1 macro), não só pela EDA:

- CRIAÇÃO: o ganho real vem de **razões** (ex.: aprovados/matriculados). Árvores
  de decisão não conseguem derivar divisões internamente, então essas features
  agregam informação. Já somas e diferenças (totais, reprovações = matriculados −
  aprovados) o XGBoost reconstrói sozinho — testadas, foram redundantes/ruído e
  pioraram o modelo, por isso não entram.
- SELEÇÃO: as variáveis macroeconômicas/demográficas têm correlação linear ~0
  com o alvo, mas removê-las PIOROU o F1 na CV (o XGBoost extrai sinal não-linear
  delas). Por isso o padrão é NÃO descartar (drop=False). A lista abaixo fica
  documentada para experimentação.
- NOMINAIS: categorias codificadas como inteiros são marcadas como `category`
  para que o XGBoost as trate de forma nativa (e não como números ordenados).
"""
import numpy as np
import pandas as pd

# Correlação linear ~0 com o alvo (EDA), MAS removê-las piora o modelo na CV.
# Mantidas por padrão; lista preservada apenas para experimentação (drop=True).
DROP_FEATURES = [
    "Nacionality",
    "International",
    "Educational special needs",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Categorias nominais (sem ordem) codificadas como inteiros. Marcadas como
# `category` para o XGBoost não assumir ordem entre os códigos.
# Binárias (0/1) ficam de fora de propósito: para 2 valores, corte numérico
# e categórico são equivalentes.
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
    foram testadas e não melhoraram (ver histórico/docstring do módulo).
    """
    df = df.copy()

    e1 = df["Curricular units 1st sem (enrolled)"]
    a1 = df["Curricular units 1st sem (approved)"]
    e2 = df["Curricular units 2nd sem (enrolled)"]
    a2 = df["Curricular units 2nd sem (approved)"]

    df["approval_rate_2nd"] = _safe_ratio(a2, e2)
    df["overall_approval_rate"] = _safe_ratio(a1 + a2, e1 + e2)

    return df


def prepare(
    df: pd.DataFrame,
    drop: bool = False,
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
