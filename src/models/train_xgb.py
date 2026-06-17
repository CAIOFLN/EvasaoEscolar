"""Treino de um classificador XGBoost para previsão de evasão escolar.

Aplica a seleção/engenharia de atributos de src.features.build_features e
trata as variáveis nominais nativamente (enable_categorical).

Uso:
    python -m src.models.train_xgb           # treino padrão (hiperparâmetros fixos)
    python -m src.models.train_xgb --tune    # otimiza hiperparâmetros via GridSearchCV
"""
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.load import RAW_CSV, TARGET, load_raw
from src.features.build_features import prepare
from src.models.evaluate import evaluate, report
from src.models.persistence import save_model

PROJECT_ROOT = RAW_CSV.resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
RANDOM_STATE = 42

# Grade de hiperparâmetros para o GridSearchCV (36 combinações x 5 folds)
PARAM_GRID = {
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [300, 500],
    "subsample": [0.8, 1.0],
}


def _base_estimator(n_jobs: int = -1) -> XGBClassifier:
    """Configuração comum (sem os hiperparâmetros que o grid search varia)."""
    return XGBClassifier(
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        enable_categorical=True,   # trata colunas 'category' nativamente
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def build_model() -> XGBClassifier:
    # Melhores hiperparâmetros encontrados pelo GridSearchCV (--tune), F1 macro.
    model = _base_estimator()
    model.set_params(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8)
    return model


def tune_model(X_train, y_train, verbose: bool = True) -> XGBClassifier:
    """Otimiza hiperparâmetros com GridSearchCV (CV estratificada, F1 macro)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs=1 no estimador para o paralelismo ficar a cargo do GridSearchCV
    search = GridSearchCV(
        _base_estimator(n_jobs=1),
        PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    search.fit(X_train, y_train)
    if verbose:
        print("\nMelhores hiperparâmetros:", search.best_params_)
        print(f"Melhor F1 macro (CV): {search.best_score_:.4f}")
    return search.best_estimator_


def get_xy(add_features: bool):
    """Prepara X (com/sem engenharia) e y codificado."""
    df = prepare(load_raw(), add_features=add_features)
    X = df.drop(columns=[TARGET])
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    return X, y, le


def run(add_features: bool, label: str, verbose: bool = True, tune: bool = False):
    X, y, le = get_xy(add_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    if tune:
        model = tune_model(X_train, y_train, verbose=verbose)
    else:
        model = build_model()
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    if verbose:
        nomes = le.inverse_transform(sorted(np.unique(y)))
        print(f"\n=== {label} | {X.shape[1]} features ===")
        print({k: round(v, 4) for k, v in metrics.items()})
        print("\nRelatório por classe:\n", report(
            le.inverse_transform(y_test), le.inverse_transform(y_pred)))
        print("Matriz de confusão (linhas=real, colunas=previsto):")
        print("classes:", list(nomes))
        print(confusion_matrix(y_test, y_pred))
    return model, X, le, metrics, y_test, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino XGBoost p/ evasão escolar.")
    parser.add_argument("--tune", action="store_true",
                        help="otimiza hiperparâmetros com GridSearchCV")
    args = parser.parse_args()

    label = "COM razões de aprovação" + (" + GridSearch" if args.tune else "")
    model, X, le, eng, _, _ = run(add_features=True, label=label, tune=args.tune, verbose=True)
    print({"ENGENHARIA": {k: round(v, 4) for k, v in eng.items()}})

    # Importância das features (ganho)
    importances = model.get_booster().get_score(importance_type="gain")
    importances = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    print("\nTop 15 features por importância (gain):")
    for name, val in importances[:15]:
        print(f"  {name:45s} {val:8.2f}")

    save_model(model, le, X.columns, MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
