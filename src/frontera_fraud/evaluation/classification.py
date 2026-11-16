from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
