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

# Rótulos em português para exibição (gráficos/relatórios). Inclui as features
# criadas em engineer(). Use FEATURE_LABELS_PT.get(col, col) para ter fallback.
# UC = Unidades Curriculares.
FEATURE_LABELS_PT = {
    "Marital Status": "Estado civil",
    "Application mode": "Modo de candidatura",
    "Application order": "Ordem de candidatura",
    "Course": "Curso",
    "Daytime/evening attendance": "Turno (diurno/noturno)",
    "Previous qualification": "Qualificação anterior",
    "Previous qualification (grade)": "Nota da qualificação anterior",
    "Nacionality": "Nacionalidade",
    "Mother's qualification": "Escolaridade da mãe",
    "Father's qualification": "Escolaridade do pai",
    "Mother's occupation": "Ocupação da mãe",
    "Father's occupation": "Ocupação do pai",
    "Admission grade": "Nota de admissão",
    "Displaced": "Deslocado (mora fora)",
    "Educational special needs": "Necessidades educacionais especiais",
    "Debtor": "Devedor",
    "Tuition fees up to date": "Mensalidade em dia",
    "Gender": "Gênero",
    "Scholarship holder": "Bolsista",
    "Age at enrollment": "Idade na matrícula",
    "International": "Internacional",
    "Curricular units 1st sem (credited)": "UC 1º sem (creditadas)",
    "Curricular units 1st sem (enrolled)": "UC 1º sem (matriculadas)",
    "Curricular units 1st sem (evaluations)": "UC 1º sem (avaliações)",
    "Curricular units 1st sem (approved)": "UC 1º sem (aprovadas)",
    "Curricular units 1st sem (grade)": "UC 1º sem (nota)",
    "Curricular units 1st sem (without evaluations)": "UC 1º sem (sem avaliação)",
    "Curricular units 2nd sem (credited)": "UC 2º sem (creditadas)",
    "Curricular units 2nd sem (enrolled)": "UC 2º sem (matriculadas)",
    "Curricular units 2nd sem (evaluations)": "UC 2º sem (avaliações)",
    "Curricular units 2nd sem (approved)": "UC 2º sem (aprovadas)",
    "Curricular units 2nd sem (grade)": "UC 2º sem (nota)",
    "Curricular units 2nd sem (without evaluations)": "UC 2º sem (sem avaliação)",
    "Unemployment rate": "Taxa de desemprego",
    "Inflation rate": "Taxa de inflação",
    "GDP": "PIB",
    "approval_rate_2nd": "Taxa de aprovação (2º sem)",
    "overall_approval_rate": "Taxa de aprovação (geral)",
}


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
