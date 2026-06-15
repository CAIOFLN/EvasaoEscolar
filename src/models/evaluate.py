"""Métricas de avaliação do classificador."""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, average: str = "macro") -> dict:
    """Retorna acurácia, precisão, recall e F1 (média macro por padrão)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def report(y_true, y_pred) -> str:
    """Relatório detalhado por classe."""
    return classification_report(y_true, y_pred, zero_division=0)


def confusion(y_true, y_pred):
    """Matriz de confusão."""
    return confusion_matrix(y_true, y_pred)
