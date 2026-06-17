"""Treino de um classificador KNN para previsão de evasão escolar.

O KNN baseia-se em distâncias, então o pré-processamento é decisivo:

- ESCALA: todas as features numéricas/binárias são padronizadas (StandardScaler).
  Sem isso, atributos de magnitude grande (ex.: códigos de curso) dominariam a
  distância e o KNN ficaria inútil.
- CATEGÓRICAS NOMINAIS: codificadas como inteiros no dataset, mas sem ordem.
  Tratá-las como números distorce a distância, então são aplicadas via
  OneHotEncoder (cada categoria vira uma dimensão binária).
- DIMENSIONALIDADE: o one-hot infla muito o número de colunas e o KNN sofre com
  a "maldição da dimensionalidade". Por isso comparamos KNN COM e SEM PCA — o
  GridSearchCV escolhe, por F1 macro, se a redução ajuda e com quantos componentes.

Uso:
    python -m src.models.train_knn           # busca padrão (grade reduzida)
    python -m src.models.train_knn --tune    # grade ampla de hiperparâmetros
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.data.load import RAW_CSV, TARGET, load_raw
from src.features.build_features import NOMINAL_FEATURES, prepare
from src.models.evaluate import evaluate, report
from src.models.persistence import save_model

PROJECT_ROOT = RAW_CSV.resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "knn.joblib"
RANDOM_STATE = 42


## A sacada é que durante o grid search ele muda o conteudo de reduce, colocando ou tirando o PCA
PARAM_GRID = [
    {  # SEM redução de dimensionalidade
        "reduce": ["passthrough"],
        "knn__n_neighbors": [15, 25, 35],
        "knn__weights": ["uniform", "distance"],
    },
    {  # COM PCA
        "reduce": [PCA(random_state=RANDOM_STATE)],
        "reduce__n_components": [10, 20, 30],
        "knn__n_neighbors": [15, 25, 35],
        "knn__weights": ["uniform", "distance"],
    },
]

# Grade mais ampla que fizemos depois
PARAM_GRID_TUNE = [
    {
        "reduce": ["passthrough"],
        "knn__n_neighbors": [5, 9, 15, 21, 25, 31, 35, 41],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    },
    {
        "reduce": [PCA(random_state=RANDOM_STATE)],
        "reduce__n_components": [5, 10, 15, 20, 25, 30, 40],
        "knn__n_neighbors": [5, 9, 15, 21, 25, 31, 35, 41],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    },
]


def build_pipeline(X) -> Pipeline:
    """Pipeline: pré-processa -> (PCA opcional) -> KNN.

    As nominais entram via one-hot; o restante (numéricas + binárias) é
    padronizado. O passo `reduce` é um placeholder que o GridSearchCV troca
    entre "passthrough" (sem PCA) e PCA com nº de componentes variável.
    """
    nominal = [c for c in NOMINAL_FEATURES if c in X.columns]
    numeric = [c for c in X.columns if c not in nominal]

    pre = ColumnTransformer(
        transformers=[
            # saída densa: o PCA não lida bem com matrizes esparsas (centragem)
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("scale", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        steps=[
            ("pre", pre),
            ("reduce", "passthrough"),
            ("knn", KNeighborsClassifier()),
        ]
    )


def get_xy(add_features: bool):
    """Prepara X e y codificado (mesmo split dos demais modelos)."""
    df = prepare(load_raw(), add_features=add_features)
    # O KNN não usa o dtype `category`; o ColumnTransformer cuida da codificação.
    X = df.drop(columns=[TARGET]).copy()
    for col in X.columns:
        if str(X[col].dtype) == "category":
            X[col] = X[col].astype(int)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return X, y, le


def _summarize_pca(cv_results, with_pca: bool) -> tuple:
    """Melhor (F1 macro, params) entre os candidatos com/sem PCA."""
    best_score, best_params = -1.0, None
    for score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        is_pca = not isinstance(params["reduce"], str)  # str == "passthrough"
        if is_pca == with_pca and score > best_score:
            best_score, best_params = score, params
    return best_score, best_params


def run(add_features: bool, label: str, tune: bool = False, verbose: bool = True):
    X, y, le = get_xy(add_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(X),
        PARAM_GRID_TUNE if tune else PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    if verbose:
        no_pca = _summarize_pca(search.cv_results_, with_pca=False)
        com_pca = _summarize_pca(search.cv_results_, with_pca=True)
        print(f"\n=== {label} | {X.shape[1]} features ===")
        print("Comparação na validação cruzada (F1 macro):")
        print(f"  SEM PCA : {no_pca[0]:.4f}  ->  "
              f"k={no_pca[1]['knn__n_neighbors']}, weights={no_pca[1]['knn__weights']}")
        print(f"  COM PCA : {com_pca[0]:.4f}  ->  "
              f"n_components={com_pca[1].get('reduce__n_components')}, "
              f"k={com_pca[1]['knn__n_neighbors']}, weights={com_pca[1]['knn__weights']}")
        vencedor = "COM PCA" if com_pca[0] >= no_pca[0] else "SEM PCA"
        print(f"  -> Melhor configuração: {vencedor}")
        print("\nMétricas no conjunto de teste (melhor modelo):")
        print({k: round(v, 4) for k, v in metrics.items()})
        nomes = le.inverse_transform(sorted(np.unique(y)))
        print("\nRelatório por classe:\n", report(
            le.inverse_transform(y_test), le.inverse_transform(y_pred)))
        print("Matriz de confusão (linhas=real, colunas=previsto):")
        print("classes:", list(nomes))
        print(confusion_matrix(y_test, y_pred))

    return model, X, le, metrics, y_test, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino KNN p/ evasão escolar.")
    parser.add_argument("--tune", action="store_true",
                        help="usa a grade ampla de hiperparâmetros")
    args = parser.parse_args()

    model, X, le, eng, _, _ = run(add_features=True, label="COM razões de aprovação",
                                  tune=args.tune, verbose=True)
    print({"ENGENHARIA": {k: round(v, 4) for k, v in eng.items()}})

    save_model(model, le, X.columns, MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
