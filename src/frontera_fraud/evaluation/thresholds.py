from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .classification import classification_metrics


def sweep_thresholds(
    y_true: np.ndarray,
    risk_percentile: np.ndarray,
    percentiles: Sequence[float] = (0.90, 0.95, 0.97, 0.98, 0.99),
) -> pd.DataFrame:
    """Calcula métricas para distintos percentiles de alerta.

    Una transacción es alerta si su risk_percentile es >= al percentil evaluado.
    """
    y_true = np.asarray(y_true)
    risk = np.asarray(risk_percentile, dtype=float)

    rows = []
    for p in percentiles:
        y_pred = (risk >= p).astype(int)
        m = classification_metrics(y_true, y_pred)
        rows.append(
            {
                "percentil_alerta": p,
                "alertas": int(y_pred.sum()),
                "precision": m["precision"],
                "recall": m["recall"],
                "fpr": m["false_positive_rate"],
                "f1": m["f1"],
            }
        )
    return pd.DataFrame(rows)
