"""Treino de um modelo baseline de classificação de evasão.

Uso:
    python -m src.models.train
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import RAW_CSV, load_raw, train_test
from src.models.evaluate import evaluate, report

PROJECT_ROOT = RAW_CSV.resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_rf.joblib"


def build_model() -> Pipeline:
    """Pipeline baseline: padronização + Random Forest.

    Todas as features do dataset são numéricas, então um único scaler basta.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def main() -> None:
    df = load_raw()
    X_train, X_test, y_train, y_test = train_test(df)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Métricas (macro):", evaluate(y_test, y_pred))
    print("\nRelatório por classe:\n", report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
