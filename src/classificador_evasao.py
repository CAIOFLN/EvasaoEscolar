from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo


DATASET_ID = 697
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class DatasetBundle:
    features: object
    target: object


def _normalizar_target(y: object) -> object:
    if getattr(y, "ndim", 1) > 1:
        y = y.iloc[:, 0]
    return y


def carregar_dados(data_path: str | None = None, target_column: str = "Target") -> DatasetBundle:
    if data_path:
        dataset = pd.read_csv(data_path)
        if target_column not in dataset.columns:
            raise ValueError(f"Coluna alvo '{target_column}' não encontrada em {data_path}.")

        x = dataset.drop(columns=[target_column]).copy()
        y = dataset[target_column].copy()
        return DatasetBundle(features=x, target=_normalizar_target(y))

    dataset = fetch_ucirepo(id=DATASET_ID)
    x = dataset.data.features.copy()
    y = _normalizar_target(dataset.data.targets.copy())

    return DatasetBundle(features=x, target=y)


def analise_inicial(dados: DatasetBundle) -> None:
    x = dados.features
    y = dados.target

    print("=== Etapa 1: Análise inicial dos dados ===")
    print(f"Amostras: {x.shape[0]}")
    print(f"Features: {x.shape[1]}")
    print(f"Valores faltantes totais: {int(x.isna().sum().sum())}")
    print("Distribuição da variável-alvo:")
    print(y.value_counts(normalize=True).sort_index().map(lambda v: f"{v:.2%}"))
    print()


def montar_pipeline(dados: DatasetBundle) -> Pipeline:
    x = dados.features

    categorical_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))],
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    print("=== Etapa 2: Feature engineering ===")
    print(f"Features numéricas: {len(numeric_cols)}")
    print(f"Features categóricas: {len(categorical_cols)}")
    print("Imputação e one-hot encoding configurados.")
    print()

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessador", preprocessador), ("modelo", modelo)])


def treinar_e_avaliar(dados: DatasetBundle, pipeline: Pipeline) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        dados.features,
        dados.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dados.target,
    )

    pipeline.fit(x_train, y_train)
    predicoes = pipeline.predict(x_test)

    print("=== Etapa 3: Treinamento do modelo ===")
    print(f"Acurácia: {accuracy_score(y_test, predicoes):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, predicoes):.4f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, predicoes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificador de evasão/resultado acadêmico.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Caminho opcional para CSV local (usa dataset UCI por padrão).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Target",
        help="Nome da coluna alvo quando usar --data-path.",
    )
    args = parser.parse_args()

    dados = carregar_dados(data_path=args.data_path, target_column=args.target_column)
    analise_inicial(dados)
    pipeline = montar_pipeline(dados)
    treinar_e_avaliar(dados, pipeline)


if __name__ == "__main__":
    main()
