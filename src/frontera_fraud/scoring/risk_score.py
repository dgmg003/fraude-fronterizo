from __future__ import annotations

from enum import Enum

import numpy as np


class RiskLevel(str, Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"


def classify_risk(
    risk_percentile: np.ndarray,
    review_threshold: float,
    alert_threshold: float,
) -> np.ndarray:
    risk = np.asarray(risk_percentile, dtype=float)
    return np.where(
        risk >= alert_threshold,
        RiskLevel.ALTO.value,
        np.where(risk >= review_threshold, RiskLevel.MEDIO.value, RiskLevel.BAJO.value),
    )
