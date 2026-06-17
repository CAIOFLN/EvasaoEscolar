"""Salvar e recarregar os modelos treinados (.joblib em models/).

Centraliza o formato do bundle salvo para que os scripts de treino
(src/models/train_*.py) e os notebooks usem exatamente a mesma estrutura:

    {"model": <estimador>, "label_encoder": <LabelEncoder>, "features": [...]}
"""
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split


def save_model(model, label_encoder, features, path: Path) -> Path:
    """Persiste o modelo treinado (+ encoder e nomes das features) em `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "label_encoder": label_encoder, "features": list(features)},
        path,
    )
    return path


def load_bundle(path: Path) -> dict:
    """Carrega o bundle salvo. Erro claro se o modelo ainda não foi treinado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Treine o modelo antes, ex.: "
            f"python -m src.models.train_xgb"
        )
    return joblib.load(path)


def test_predictions(get_xy, model_path: Path, add_features: bool = True,
                     test_size: float = 0.2, random_state: int = 42):
    """Carrega o modelo salvo e avalia no MESMO split de teste do treino.

    Reproduz o split estratificado determinístico (`random_state` fixo) usado
    nos scripts e devolve (label_encoder, y_test, y_pred) — tudo em rótulos
    codificados, prontos para `evaluate`/`inverse_transform`.

    `get_xy` é a função homônima do módulo de treino (cada modelo prepara X de
    um jeito; usamos a do próprio modelo para casar com o que foi treinado).
    """
    bundle = load_bundle(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]

    X, y, _ = get_xy(add_features)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    y_pred = model.predict(X_test)
    return le, y_test, y_pred
